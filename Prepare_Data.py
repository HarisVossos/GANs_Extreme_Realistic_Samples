### Climate_Data Preprocessing ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry_convert as pc
import requests
from bs4 import BeautifulSoup
import os
cur_dir = os.getcwd()
print("Current Directory : " , cur_dir)

dir = cur_dir  + '\data\climate_data.xlsx'
climate_data = pd.read_excel(dir)
columns_to_keep = ['Year','Country',"Total Damages, Adjusted ('000 US$)",'ISO']
new_climate_data = climate_data[columns_to_keep]
new_climate_data = new_climate_data.rename(columns = {"Total Damages, Adjusted ('000 US$)":"Losses"})

indices = new_climate_data[new_climate_data.ISO == 'DFR'].index
for x in list(indices):
    new_climate_data.at[x,'ISO'] = 'DEU'

indices = new_climate_data[new_climate_data.ISO == 'AZO'].index
for x in list(indices):
    new_climate_data.at[x,'ISO'] = 'PRT'

invalid_ISO = []
def custom_function_alpha3_to_alpha2(x):
    try:
        pc.country_alpha3_to_country_alpha2(x)
        return pc.country_alpha3_to_country_alpha2(x)
    except KeyError:
        invalid_ISO.append(x)
        return 'Invalid'


def custom_function_alpha2_to_continent(x):
    try:
        pc.country_alpha2_to_continent_code(x)
        return pc.country_alpha2_to_continent_code(x)

    except KeyError:
        return 'Invalid'

invalid_ISO = []
new_climate_data['Alpha_2'] = new_climate_data['ISO'].apply(custom_function_alpha3_to_alpha2)
new_climate_data['Continent'] = new_climate_data['Alpha_2'].apply(custom_function_alpha2_to_continent)
new_climate_data = new_climate_data.dropna()

#### Zones ####

page = BeautifulSoup(requests.get("https://developers.google.com/public-data/docs/canonical/countries_csv").content, 'lxml')
table = page.find('table')
data = table.find_all('tr')[1:]
cntry_lat_df = pd.DataFrame(columns=["ISO2 CODE", "Latitude", "Longitude", "Country"])
# cntry_lat_df = cntry_lat_df.dropna(how='any')
for row in data:
    lst = row.find_all('td')
    code = lst[0].text
    # Latitude
    try:
        lat = float(lst[1].text)
    except:
        continue
    # Longitude
    try:
        lon = float(lst[2].text)
    except:
        continue
    # Country
    cntry = lst[3].text
    cntry_lat_df.loc[len(cntry_lat_df.index)] = [code, lat, lon, cntry]


country_codes = pd.read_excel(cur_dir + '\data\Country_Codes.xlsx')
cntry_lat_df2 = cntry_lat_df.merge(country_codes,how='left', left_on='ISO2 CODE',right_on = 'ISO2 CODE').copy()
cntry_lat_df2 = cntry_lat_df2[["ISO3 CODE", "Latitude", "Longitude", "Country"]]
missing_lat_lon_df = pd.DataFrame({
                        'ISO3 CODE': ['FSM', 'NAM', 'MAF','SXM'],
                        'Latitude': [7.4256, -22.9576, 18.0708, 18.0708 ],
                        'Longitude': [150.5508, 18.4904, -63.0501, -63.0501],
                        'Country': ['Micronesia', 'Namibia', 'St. Martin (French part)',
                                   'Sint Maarten (Dutch part)']
                        })
cntry_lat_df3 = pd.concat([cntry_lat_df2, missing_lat_lon_df], ignore_index=True).sort_values('ISO3 CODE').copy()


def latitude_zone(df):
    if (abs(df['Latitude']) >= 0) and (abs(df['Latitude']) <= 23.5):
        return 'Tropical'

    elif (abs(df['Latitude']) > 23.5) and (abs(df['Latitude']) <= 40):
        return 'Subtropical'

    elif (abs(df['Latitude']) > 40) and (abs(df['Latitude']) <= 60):
        return 'Temperate'

    elif (abs(df['Latitude']) > 60) and (abs(df['Latitude']) <= 90):
        return 'Polar'
    else:
        return "no"

cntry_lat_df3['Zone'] = cntry_lat_df3.apply(latitude_zone, axis=1)


def custom_merge(iso):
    if len(cntry_lat_df3.loc[cntry_lat_df3['ISO3 CODE'] == iso, 'Zone'].values) != 0:
        return cntry_lat_df3.loc[cntry_lat_df3['ISO3 CODE'] == iso, 'Zone'].values[0]
    else:
        return 'Not_Found'

new_climate_data['Zone'] = new_climate_data['ISO'].apply(lambda x: custom_merge(x))
drop_index = new_climate_data[new_climate_data.Zone == 'Not_Found'].index
new_climate_data = new_climate_data.drop(drop_index)
zones = list(new_climate_data.Zone.unique())
data = []
for zone in zones:
    group_data = new_climate_data[new_climate_data.Zone == zone]
    data.append(group_data)


Tropical = data[0]
Temperate = data[1]
Subtropical = data[2]
Polar = data[3]

### save data to data folder in the current directory as csv files ###
Tropical.to_csv(cur_dir + '\data\Tropical_dataset.csv')
Temperate.to_csv(cur_dir + '\data\Temperate_dataset.csv')
Subtropical.to_csv(cur_dir + '\data\Subtropical_dataset.csv')
Polar.to_csv(cur_dir + '\data\Polar_dataset.csv')