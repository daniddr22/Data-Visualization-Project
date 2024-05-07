import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.cluster import DBSCAN

def load_data():        ### PATHS TO REPLACE
    unicorn_df= pd.read_excel('/Users/fab/Desktop/UNICORN_ANALYSIS/data_viz_presentation/INFO_UNICORN_FINAL.xlsx')
    vcs= pd.read_excel('/Users/fab/Desktop/UNICORN_ANALYSIS/data_viz_presentation/INFO_INVESTORS.xlsx')
    capital= pd.read_csv('/Users/fab/Desktop/UNICORN_ANALYSIS/data_viz_presentation/country-capital-lat-long-population.csv')
    countries_df = pd.read_excel('/Users/fab/Desktop/UNICORN_ANALYSIS/data_viz_presentation/Startup_per_country.xlsx', skiprows=3)
    unicorn_df2= pd.read_excel('/Users/fab/Desktop/UNICORN_ANALYSIS/data_viz_presentation/INFO_UNICORN_FINAL.xlsx')
    vcs2= pd.read_excel('/Users/fab/Desktop/UNICORN_ANALYSIS/data_viz_presentation/INFO_INVESTORS.xlsx')
    capital2= pd.read_csv('/Users/fab/Desktop/UNICORN_ANALYSIS/data_viz_presentation/country-capital-lat-long-population.csv')
    countries_df2 = pd.read_excel('/Users/fab/Desktop/UNICORN_ANALYSIS/data_viz_presentation/Startup_per_country.xlsx', skiprows=3)
    return unicorn_df, vcs, capital, countries_df, unicorn_df2, vcs2, capital2, countries_df2

def preprocess_data(unicorn_df, vcs, capital, countries_df):
        ### mappa daniele
    # Filtraggio delle righe con "venture_capital"
    df_vc = vcs[vcs['TYPE'].str.contains('venture_capital', case=False)]
    # Selezione delle colonne rilevanti, incluso "HQ CITY"
    df_vc_filtered = df_vc[['NAME', 'LATITUDE', 'LONGITUDE', 'LAUNCH YEAR', 'HQ CITY']]
    # Aggiungi una colonna 'ID' con valore 0 a df_vc_filtered
    df_vc_filtered = df_vc_filtered.assign(ID=0)
    # Rimozione delle righe con "LAUNCH YEAR" sotto il 1980
    df_vc_filtered = df_vc_filtered[df_vc_filtered['LAUNCH YEAR'] >= 1980]
    # Convertendo l'anno di lancio in interi
    df_vc_filtered['LAUNCH YEAR'] = df_vc_filtered['LAUNCH YEAR'].astype(int)
    # Sostituzione dei valori NaN con uno spazio vuoto nella colonna "HQ CITY"
    df_vc_filtered['HQ CITY'] = df_vc_filtered['HQ CITY'].fillna(' ')
    # Estensione del dataset per mantenere i puntini visibili
    # Creando un nuovo dataset con un record per ogni VC per ogni anno dal lancio ad oggi
    max_year = df_vc_filtered['LAUNCH YEAR'].max()
    df_vc_extended = pd.concat(
        [df_vc_filtered[df_vc_filtered['LAUNCH YEAR'] <= year].assign(**{'LAUNCH YEAR': year}) for year in range(1980, max_year+1)],
        ignore_index=True
    )
    

    # Selezione delle colonne rilevanti, incluso "HQ CITY"
    df_un_filtered = unicorn_df[['NAME', 'LATITUDE', 'LONGITUDE', 'LAUNCH YEAR', 'HQ CITY']]
    # Aggiungi una colonna 'ID' con valore 0 a df_vc_filtered
    df_un_filtered = df_un_filtered.assign(ID=1)
    # Rimuovere le righe con valori non validi nella colonna "LAUNCH YEAR"
    df_un_filtered = df_un_filtered.dropna(subset=['LAUNCH YEAR'])
    # Convertire l'anno di lancio in interi
    df_un_filtered['LAUNCH YEAR'] = df_un_filtered['LAUNCH YEAR'].astype(int)
    # Sostituzione dei valori NaN con uno spazio vuoto nella colonna "HQ CITY"
    df_un_filtered['HQ CITY'] = df_un_filtered['HQ CITY'].fillna(' ')
    # Estensione del dataset per mantenere i puntini visibili
    # Creando un nuovo dataset con un record per ogni Unicorn per ogni anno dal lancio ad oggi
    max_year = df_un_filtered['LAUNCH YEAR'].max()
    df_un_extended = pd.concat(
        [df_un_filtered[df_un_filtered['LAUNCH YEAR'] <= year].assign(**{'LAUNCH YEAR': year}) for year in range(1980, max_year+1)],
        ignore_index=True
    )
    

    unicorn_df.drop(
        columns=[
            'PROFILE URL', 'WEBSITE', 'TAGLINE', 'ADDRESS', 'STREET', 'STREET NUMBER', 
            'STREET AND STREET NUMBER', 'TEAM (DEALROOM)', 'TEAM (EDITORIAL)', 
            'LAUNCH MONTH', 'LAUNCH DATE', 'CLOSING YEAR', 'CLOSING MONTH',
            'CLOSING DATE', 'DELIVERY METHOD', 'EMPLOYEES LATEST', 'LAST KPI DATE', 
            'FINANCIALS CURRENCY', 'VALUATION', 'VALUATION CURRENCY', 'VALUATION (USD)', 
            'HISTORICAL VALUATIONS - VALUES USD M', 'FACEBOOK LIKES',
            'TWITTER FOLLOWERS', 'TWITTER TWEETS', 'TWITTER FAVORITES', 
            'WEBSITE TRAFFIC ESTIMATE 6 MONTH', 'FACEBOOK LIKES DATES', 'FACEBOOK LIKES VALUES', 
            'TWITTER FOLLOWERS DATES', 'TWITTER FOLLOWERS VALUES', 'FACEBOOK', 'TWITTER', 
            'LINKEDIN', 'ANGELLIST', 'CRUNCHBASE', 'GOOGLE PLAY LINK', 'ITUNES LINK',
            'LOGO', 'FOUNDERS LINKEDIN', 'FOUNDERS STRENGTH', 'LISTS', 
            'APP DOWNLOADS LATEST ESTIMATE (IOS)', 'APP DOWNLOADS 6 MONTHS ESTIMATE (IOS)',
            'APP DOWNLOADS 12 MONTHS ESTIMATE (IOS)', 'APP DOWNLOADS LATEST ESTIMATE (ANDROID)', 
            'APP DOWNLOADS 6 MONTHS ESTIMATE (ANDROID)', 'APP DOWNLOADS 12 MONTHS ESTIMATE (ANDROID)', 
            'ALL TAGS', 'WEBSITE TRAFFIC RANK 3/6/12 MONTHS', 'EMPLOYEE RANK 3/6/12 MONTHS', 
            'APP RANK 3/6/12 MONTHS', 'TRADE REGISTER NUMBER', 'TRADE REGISTER NAME',
            'TRADE REGISTER URL', 'CORE SIDE VALUE', 'YEAR COMPANY BECAME FUTURE UNICORN',
            'PIC NUMBER', 'DEALROOM SIGNAL - RATING', 'DEALROOM SIGNAL - COMPLETENESS', 
            'DEALROOM SIGNAL - FOUNDING TEAM SCORE', 'DEALROOM SIGNAL - TIMING', 'CUSTOM HQ REGIONS', 'LOCATIONS', 'FOUNDING LOCATION',
            'LEAD INVESTORS', 'SDGS', 'ZIPCODE'
        ],
        inplace=True
    )
    
    ### PREPROCESSING DAVID 2
    founders = unicorn_df[['NAME','HQ REGION','TOTAL FUNDING (EUR M)', 'LAUNCH YEAR','FOUNDERS GENDERS',
                        'FOUNDERS IS SERIAL','FOUNDERS BACKGROUNDS','FOUNDERS UNIVERSITIES',
                        'FOUNDERS COMPANY EXPERIENCE','FOUNDERS EXPERIENCE','FOUNDERS FOUNDED COMPANIES TOTAL FUNDING',
                        'FOUNDERS YEARS OF EDUCATION','IS FOUNDERS FIRST COMPANY', 'YEAR COMPANY BECAME UNICORN']].copy()

    split = ['FOUNDERS GENDERS', 'FOUNDERS IS SERIAL', 'FOUNDERS BACKGROUNDS', 'FOUNDERS UNIVERSITIES',
             'FOUNDERS COMPANY EXPERIENCE', 'FOUNDERS EXPERIENCE', 'FOUNDERS FOUNDED COMPANIES TOTAL FUNDING',
             'FOUNDERS YEARS OF EDUCATION', 'IS FOUNDERS FIRST COMPANY']

    for col in split:
        founders[col] = founders[col].apply(lambda x: x.split(';') if isinstance(x, str) else (x.tolist() if hasattr(x, 'tolist') else []))

    ### Education
    edu_cols = ['LAUNCH YEAR','TOTAL FUNDING (EUR M)', 'FOUNDERS BACKGROUNDS', 'FOUNDERS UNIVERSITIES', 'FOUNDERS YEARS OF EDUCATION','YEAR COMPANY BECAME UNICORN']
    founders_edu = founders[['LAUNCH YEAR','TOTAL FUNDING (EUR M)','FOUNDERS BACKGROUNDS', 'FOUNDERS UNIVERSITIES', 'FOUNDERS YEARS OF EDUCATION','YEAR COMPANY BECAME UNICORN']].copy()
    for col in edu_cols:
        founders_edu = founders_edu.explode(col)

    for col in edu_cols:
        founders_edu[col] = founders_edu[col].replace('n/a', np.nan)
    founders_edu.reset_index(drop = True)
    founders_edu['FOUNDERS UNIVERSITIES'] = founders_edu['FOUNDERS UNIVERSITIES'].apply(lambda x: x.split(',') if isinstance(x, str) else (x.tolist() if hasattr(x, 'tolist') else []))
    founders_edu['FOUNDERS BACKGROUNDS'] = founders_edu['FOUNDERS BACKGROUNDS'].apply(lambda x: x.split(',') if isinstance(x, str) else (x.tolist() if hasattr(x, 'tolist') else []))
    
    
    #EVOLUZIONE VC E UNICORN
    unicorns_per_year = unicorn_df.groupby('LAUNCH YEAR')['NAME'].count()
    unicorns_per_year= unicorns_per_year.to_frame()
    unicorns_per_year= unicorns_per_year.reset_index()
    unicorns_per_year.rename(columns={'NAME': 'NUMBER OF UNICORNS FOUNDED'}, inplace=True)
    unicorns_per_year['Cumulative Unicorn FOUNDED'] = unicorns_per_year['NUMBER OF UNICORNS FOUNDED'].cumsum()
    unicorns_per_year['LAUNCH YEAR']= pd.to_datetime(unicorns_per_year['LAUNCH YEAR'], format='%Y')
    
    vc_per_year= vcs.groupby('LAUNCH YEAR')['NAME'].count()
    vc_per_year= vc_per_year.to_frame()
    vc_per_year= vc_per_year.reset_index()
    vc_per_year.rename(columns={'NAME': 'NUMBER OF VC FOUNDED'}, inplace=True)
    vc_per_year = vc_per_year[vc_per_year['LAUNCH YEAR'] >= 1980]
    vc_per_year['Cumulative VC Founded'] = vc_per_year['NUMBER OF VC FOUNDED'].cumsum()
    vc_per_year['LAUNCH YEAR']= pd.to_datetime(vc_per_year['LAUNCH YEAR'], format='%Y')

    unicorn_and_vc_per_year= pd.merge(vc_per_year, unicorns_per_year, on='LAUNCH YEAR', how='outer')
    unicorn_and_vc_per_year['LAUNCH YEAR'] = pd.to_datetime(unicorn_and_vc_per_year['LAUNCH YEAR']).dt.year
    unicorn_and_vc_per_year.fillna(method='ffill', inplace=True)
    unicorn_and_vc_per_year= unicorn_and_vc_per_year.fillna(0)
    
    
    ## TEMPO NECESSARIO PER DIVENTARE UNICORN
    time_needed= unicorn_df[['NAME', 'LAUNCH YEAR', 'YEAR COMPANY BECAME UNICORN', 'INDUSTRIES']]
    time_needed['LAUNCH YEAR']= pd.to_datetime(time_needed['LAUNCH YEAR'], format= '%Y')
    time_needed['YEAR COMPANY BECAME UNICORN']= pd.to_datetime(time_needed['YEAR COMPANY BECAME UNICORN'], format= '%Y')
    time_needed['LAUNCH YEAR']= time_needed['LAUNCH YEAR'].dt.year
    time_needed['YEAR COMPANY BECAME UNICORN']= time_needed['YEAR COMPANY BECAME UNICORN'].dt.year
    time_needed['time_needed']= time_needed['YEAR COMPANY BECAME UNICORN']- time_needed['LAUNCH YEAR']
    time_needed['INDUSTRIES']= time_needed['INDUSTRIES'].astype(str).str.split(';').tolist()
    time_needed= time_needed.explode('INDUSTRIES')

    industry_mapping = {
        'enterprise software': 'Technology & Software',
        'health': 'Health & Wellness',
        'transportation': 'Transportation & Logistics',
        'security': 'Technology & Software',
        'energy': 'Energy & Engineering',
        'media': 'Media & Entertainment',
        'robotics': 'Robotics & Advanced Manufacturing',
        'real estate': 'Finance & Real Estate',
        'fintech': 'Finance & Real Estate',
        'jobs recruitment': 'Human Resources & Education',
        'marketing': 'Consumer Goods & Services',
        'semiconductors': 'Technology & Software',
        'food': 'Consumer Goods & Services',
        'home living': 'Consumer Goods & Services',
        'fashion': 'Consumer Goods & Services',
        'wellness beauty': 'Health & Wellness',
        'gaming': 'Media & Entertainment',
        'service provider': 'Technology & Software',
        'travel': 'Consumer Goods & Services',
        'sports': 'Consumer Goods & Services',
        'hosting': 'Technology & Software',
        'event tech': 'Media & Entertainment',
        'telecom': 'Technology & Software',
        'space': 'Transportation & Logistics',
        'kids': 'Social & Lifestyle',
        'engineering and manufacturing equipment': 'Energy & Engineering',
        'education': 'Human Resources & Education',
        'legal': 'Finance & Real Estate',
        'dating': 'Social & Lifestyle',
        'music': 'Media & Entertainment',
        'consumer electronics': 'Technology & Software',
        'chemicals': 'Energy & Engineering'
    }

    time_needed['INDUSTRIES']= time_needed['INDUSTRIES'].map(industry_mapping)

    average_time= time_needed.groupby(['LAUNCH YEAR', 'INDUSTRIES'])['time_needed'].mean().reset_index()


    average_time.rename(columns={'time_needed': 'average_time_needed_per_industry_per_year'}, inplace=True)
    average_time_general= time_needed.groupby(['LAUNCH YEAR'])['time_needed'].mean().reset_index()

    time_needed.dropna(inplace=True)
    
    #BAR RACE AMONG INDUSTRIES
    industries= unicorn_df[['NAME', 'LAUNCH YEAR', 'INDUSTRIES']]
    industries['LAUNCH YEAR']= pd.to_datetime(industries['LAUNCH YEAR'], format='%Y')
    industries= industries.explode('INDUSTRIES')
    industries['INDUSTRIES']= industries['INDUSTRIES'].map(industry_mapping)
    industries_counts = industries.groupby(['LAUNCH YEAR', 'INDUSTRIES']).size().reset_index(name='UNICORN COUNT')
    industries_counts = industries_counts.sort_values(by=['LAUNCH YEAR', 'INDUSTRIES'])
    industries_counts['CUMULATIVE UNICORN COUNT'] = industries_counts.groupby('INDUSTRIES')['UNICORN COUNT'].cumsum()


    years = pd.date_range(start='1984-01-01', end='2024-01-01', freq='YS')
    industries = industries_counts['INDUSTRIES'].unique()

    index = pd.MultiIndex.from_product([years, industries], names=['LAUNCH YEAR', 'INDUSTRIES'])
    full_data = pd.DataFrame(index=index).reset_index()
    merged_data = full_data.merge(industries_counts, on=['LAUNCH YEAR', 'INDUSTRIES'], how='left')


    merged_data['UNICORN COUNT'].fillna(0, inplace=True)


    merged_data.sort_values(by=['INDUSTRIES', 'LAUNCH YEAR'], inplace=True)
    merged_data['CUMULATIVE UNICORN COUNT'] = merged_data.groupby('INDUSTRIES')['UNICORN COUNT'].cumsum()
    merged_data['LAUNCH YEAR'] = merged_data['LAUNCH YEAR'].astype(str)


    if pd.api.types.is_datetime64_any_dtype(industries_counts['LAUNCH YEAR']):
        industries_counts['LAUNCH YEAR'] = industries_counts['LAUNCH YEAR'].dt.year.astype(str)
    merged_data.sort_values(by=['INDUSTRIES', 'LAUNCH YEAR'], inplace=True)

    merged_data['LAUNCH YEAR'] = merged_data['LAUNCH YEAR'].astype(str)
    
    #WOMEN IN THE UNICORN ECOSYSTEM
    unicorn_gender= unicorn_df[['LAUNCH YEAR', 'FOUNDERS GENDERS', 'HQ COUNTRY']]
    vc_genders= vcs[['LAUNCH YEAR', 'TEAM GENDERS','HQ COUNTRY']].copy()
    capital= capital[['Country', 'Latitude', 'Longitude']]
    capital['Country']= capital['Country'].replace('United States of America', 'United States')
    capital['Country']= capital['Country'].replace('Croatia', 'Czech Republic')
    capital['Country']= capital['Country'].replace('Russian Federation', 'Russia')
    capital['Country']= capital['Country'].replace('Republic of Korea', 'South Korea')
    capital['Country']= capital['Country'].replace('China, Taiwan Province of China', 'Taiwan')
    capital['Country']= capital['Country'].replace('China, Hong Kong SAR', 'Hong Kong')
    capital['Country']= capital['Country'].replace('Turkey', 'Türkiye')
    capitals_dict = {row['Country']: (row['Latitude'], row['Longitude']) for index, row in capital.iterrows()}

    def count_female(text):
        return text.lower().count('female')

    vc_genders['TEAM GENDERS']= vc_genders['TEAM GENDERS'].astype(str).str.split(';').tolist()
    vc_genders= vc_genders[vc_genders['LAUNCH YEAR'] >= 1980]
    vc_genders['LAUNCH YEAR']= pd.to_datetime(vc_genders['LAUNCH YEAR'], format= '%Y')
    vc_genders= vc_genders.explode('TEAM GENDERS')
    vc_genders= vc_genders[vc_genders['TEAM GENDERS']== 'female']


    def count_female(text):
        return text.lower().count('female')


    years = [datetime(year, 1, 1) for year in range(1980, 2024)]
    new_rows_vc = []
    for country in vc_genders['HQ COUNTRY'].unique():
        launch_years = vc_genders[vc_genders['HQ COUNTRY'] == country]['LAUNCH YEAR']
        missing_years = [year for year in years if year not in launch_years.values]
        for year in missing_years:
            new_rows_vc.append({'HQ COUNTRY': country, 'LAUNCH YEAR': year})
    vc_genders = pd.concat([vc_genders, pd.DataFrame(new_rows_vc)], ignore_index=True)
    vc_genders = vc_genders.sort_values(by=['HQ COUNTRY', 'LAUNCH YEAR']).reset_index(drop=True)
    vc_genders['latitude'] = vc_genders['HQ COUNTRY'].map(lambda x: capitals_dict[x][0] if x in capitals_dict else None)
    vc_genders['longitude'] = vc_genders['HQ COUNTRY'].map(lambda x: capitals_dict[x][1] if x in capitals_dict else None)
    vc_genders['TEAM GENDERS']= vc_genders['TEAM GENDERS'].fillna('ND')
    vc_genders['women_count']= vc_genders['TEAM GENDERS'].apply(count_female)
    vc_genders['cumulative_women_count'] = vc_genders.groupby(['longitude', 'latitude'])['women_count'].cumsum()
    vc_genders= vc_genders.rename(columns={'cumulative_women_count': 'cumulative_women_count_in_vc'})

    vc_genders = vc_genders.dropna()
    vc_genders.isnull().sum()



    unicorn_gender['FOUNDERS GENDERS']= unicorn_gender['FOUNDERS GENDERS'].astype(str).str.split(';').tolist()
    unicorn_gender['LAUNCH YEAR']= pd.to_datetime(unicorn_gender['LAUNCH YEAR'], format= '%Y')
    unicorn_gender= unicorn_gender.explode('FOUNDERS GENDERS')
    unicorn_gender= unicorn_gender.rename(columns={'FOUNDERS GENDERS': 'TEAM GENDERS'})
    unicorn_gender= unicorn_gender[unicorn_gender['TEAM GENDERS']== 'female']

    new_rows_unicorn = []
    for country in unicorn_gender['HQ COUNTRY'].unique():
        launch_years = unicorn_gender[unicorn_gender['HQ COUNTRY'] == country]['LAUNCH YEAR']
        missing_years = [year for year in years if year not in launch_years.values]
        for year in missing_years:
            new_rows_unicorn.append({'HQ COUNTRY': country, 'LAUNCH YEAR': year})

    unicorn_gender = pd.concat([unicorn_gender, pd.DataFrame(new_rows_unicorn)], ignore_index=True)
    unicorn_gender = unicorn_gender.sort_values(by=['HQ COUNTRY', 'LAUNCH YEAR']).reset_index(drop=True)
    unicorn_gender['latitude'] = unicorn_gender['HQ COUNTRY'].map(lambda x: capitals_dict[x][0] if x in capitals_dict else None)
    unicorn_gender['longitude'] = unicorn_gender['HQ COUNTRY'].map(lambda x: capitals_dict[x][1] if x in capitals_dict else None)
    unicorn_gender['TEAM GENDERS']= unicorn_gender['TEAM GENDERS'].fillna('ND')
    unicorn_gender['women_count']= unicorn_gender['TEAM GENDERS'].apply(count_female)
    unicorn_gender['cumulative_women_count'] = unicorn_gender.groupby(['longitude', 'latitude'])['women_count'].cumsum()
    unicorn_gender= unicorn_gender.rename(columns={'cumulative_women_count': 'cumulative_women_count_in_unicorn'})
    available_years = sorted(set(vc_genders['LAUNCH YEAR'].dt.year) | set(unicorn_gender['LAUNCH YEAR'].dt.year))
    
    
    ### DAVID PAQUETTE
    countries = countries_df[['Country', 'VC FUNDING', 'AMOUNT OF EXITS (2000)']].copy()
    countries['AMOUNT OF EXITS (2000)'] = countries['AMOUNT OF EXITS (2000)'].replace('-', np.nan) 
    countries = countries.dropna(subset=['AMOUNT OF EXITS (2000)'])

    def convert_to_full_number(value):
        # Remove the dollar sign and commas
        number = value.replace('$', '').replace(',', '')
        if number.endswith('t'):
            # Convert from trillion
            return float(number[:-1]) * 1_000_000_000_000
        elif number.endswith('b'):
            # Convert from billion
            return float(number[:-1]) * 1_000_000_000
        elif number.endswith('m'):
            # Convert from million
            return float(number[:-1]) * 1_000_000
        elif number.endswith('k'):
            # Convert from thousand
            return float(number[:-1]) * 1_000
        else:
            # If no letter
            return float(number)

    # Apply Functions
    countries['TOTAL VC FUNDING'] = countries['VC FUNDING'].apply(convert_to_full_number)
    countries['TOTAL EXIT AMOUNTS'] = countries['AMOUNT OF EXITS (2000)'].apply(convert_to_full_number)

    # Replace any sequence of whitespace characters with a single space and then strip leading/trailing spaces
    countries['Country'] = countries['Country'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # Correct Country
    countries['Country'] = countries['Country'].replace('Switzerlan\xadd', 'Switzerland')
    countries['Country'] = countries['Country'].replace('Netherland\xads',  'Netherlands')

    # Dict of Continents
    continent = {
        "Hungary": "Europe", "Belarus": "Europe", "Austria": "Europe", "Serbia": "Europe",
        "Switzerland": "Europe", "Germany": "Europe", "Andorra": "Europe", "Bulgaria": "Europe",
        "United Kingdom": "Europe", "France": "Europe", "Montenegro": "Europe", "Luxembourg": "Europe",
        "Italy": "Europe", "Denmark": "Europe", "Finland": "Europe", "Slovakia": "Europe",
        "Norway": "Europe", "Ireland": "Europe", "Spain": "Europe", "Malta": "Europe",
        "Ukraine": "Europe", "Croatia": "Europe", "Moldova": "Europe", "Monaco": "Europe",
        "Liechtenstein": "Europe", "Poland": "Europe", "Iceland": "Europe", "San Marino": "Europe",
        "Bosnia and Herzegovina": "Europe", "Albania": "Europe", "Lithuania": "Europe",
        "North Macedonia": "Europe", "Slovenia": "Europe", "Romania": "Europe", "Latvia": "Europe",
        "Netherlands": "Europe", "Russia": "Europe", "Estonia": "Europe", "Belgium": "Europe",
        "Czech Republic": "Europe", "Greece": "Europe", "Portugal": "Europe", "Sweden": "Europe",
        "Isle of Man": "Europe", "Faroe Islands": "Europe", "Gibraltar": "Europe",
        "United States": "North America", "China": "Asia", "Africa": "Africa", "Oceania": "Oceania",
        "South America": "South America", "Canada": "North America"
    }

    countries['Continent'] = countries['Country'].map(continent)

    countries['Total'] = countries['TOTAL VC FUNDING'] + countries['TOTAL EXIT AMOUNTS']

    def number_to_string(number):

        abs_number = abs(number)
        if abs_number >= 1_000_000_000_000:
            # Convert to trillions
            formatted_number = f"{number / 1_000_000_000_000:.2f}T"
        elif abs_number >= 1_000_000_000:
            # Convert to billions
            formatted_number = f"{number / 1_000_000_000:.2f}B"
        elif abs_number >= 1_000_000:
            # Convert to millions
            formatted_number = f"{number / 1_000_000:.2f}M"
        elif abs_number >= 1_000:
            # Convert to thousands
            formatted_number = f"{number / 1_000:.2f}K"
        else:
            # Use the original number if it's less than 1000
            formatted_number = str(number)

        # Add a dollar sign if needed or any other formatting
        return f"${formatted_number}"

    #### For VC Funding

    continent_data_vc = countries.groupby('Continent').agg({'TOTAL VC FUNDING': 'sum'}).reset_index()
    continent_data_vc = continent_data_vc.sort_values(by='TOTAL VC FUNDING', ascending=False).reset_index(drop=True)
    continent_data_vc['TOTAL FUNDING'] = continent_data_vc['TOTAL VC FUNDING'].apply(number_to_string)

    #### For Exits

    # Aggregating TOTAL EXIT AMOUNTS by Continent
    continent_data_exit = countries.groupby('Continent').agg({'TOTAL EXIT AMOUNTS': 'sum'}).reset_index()
    continent_data_exit = continent_data_exit.sort_values(by='TOTAL EXIT AMOUNTS', ascending=False).reset_index(drop=True)
    continent_data_exit['AMOUNT OF EXITS (2000)'] = continent_data_exit['TOTAL EXIT AMOUNTS'].apply(number_to_string)
    
    
    ###### FOR INVESTORS ######
    exits_investors = vcs[['NAME', 'LATITUDE', 'LONGITUDE', 'HQ REGION', 'HQ CITY', 'EXITS NAMES',
                                     'EXITS AMOUNTS']].copy()
    exits_unicorns = unicorn_df[['NAME', 'LATITUDE', 'LONGITUDE', 'HQ REGION', 'HQ CITY', 'INDUSTRIES', 'COMPANY STATUS']].copy()
    split_investors = ['EXITS NAMES', 'EXITS AMOUNTS']
    for col in split_investors:
        exits_investors[col] = exits_investors[col].apply(lambda x: x.split(';') if isinstance(x, str) else (x.tolist() if hasattr(x, 'tolist') else []))
    exits_investors_exploded = exits_investors.explode('EXITS NAMES').explode('EXITS AMOUNTS')
    exits_investors_exploded = exits_investors_exploded.reset_index()
    exits_investors_exploded['EXITS AMOUNTS'] = exits_investors_exploded['EXITS AMOUNTS'].replace('n/a', np.nan)
    exits_investors_exploded = exits_investors_exploded.dropna(subset =['EXITS AMOUNTS'])
    # Total Exits
    total_exits = vcs[['NAME', 'EXITS TOTAL (EUR M)', 'HQ COUNTRY','LATITUDE', 'LONGITUDE']].copy()

    ###### FOR UNICORN ######
    unicorns = unicorn_df[['NAME', 'LATITUDE', 'LONGITUDE', 'HQ REGION', 'HQ CITY', 'INDUSTRIES']].copy()
    unicorns['INDUSTRIES'] = unicorns['INDUSTRIES'].apply(lambda x: x.split(';') if isinstance(x, str) else (x.tolist() if hasattr(x, 'tolist') else []))
    unicorns_exploded = unicorns.explode('INDUSTRIES')
    unicorns_exploded['INDUSTRIES'] = unicorns_exploded['INDUSTRIES'].replace(industry_mapping)
    unicorns_exploded = unicorns_exploded.dropna(subset=['INDUSTRIES'])

    ###### RENAME VARIABLE ######

    exits_investors_exploded = exits_investors_exploded.rename(columns={'NAME': 'INVESTORS','HQ REGION':'INVESTOR REGION',
                                                                        'HQ COUNTRY':'INVESTOR COUNTRY', 'HQ CITY':'INVESTOR CITY',
                                                                        'LATITUDE': 'INVESTOR LATITUDE','LONGITUDE': 'INVESTOR LONGITUDE',
                                                                        'EXITS NAMES':'UNICORN NAME'})

    unicorns_exploded = unicorns_exploded.rename(columns={'NAME':'UNICORN NAME','HQ REGION':'UNICORN REGION', 'HQ COUNTRY':'UNICORN COUNTRY',
                                                          'HQ CITY':'UNICORN CITY','LATITUDE': 'UNICORN LATITUDE',
                                                          'LONGITUDE': 'UNICORN LONGITUDE'})

    ###### MERGE ######
    unicorn_exits = pd.merge(exits_investors_exploded, unicorns_exploded, how = 'inner', on = 'UNICORN NAME')
    
    ###### INDUSTRIES ######
    unicorn_exits['EXITS AMOUNTS'] = unicorn_exits['EXITS AMOUNTS'].astype(float)
    industries = unicorn_exits.groupby('INDUSTRIES').agg({'EXITS AMOUNTS':'sum'}).reset_index()
    industries = industries.sort_values(by = 'EXITS AMOUNTS', ascending = False).reset_index(drop = True)

    def number_to_string(number):
        abs_number = abs(number)
        if abs_number >= 1_000_000: 
            formatted_number = f"{number / 1_000_000:.0f}T"
        elif abs_number >= 1_000:  
            formatted_number = f"{number / 1_000:.0f}B"
        elif abs_number >= 1:  
            formatted_number = f"{number:.0f}M"
        else:
            formatted_number = str(number) 

        return f"${formatted_number}"
    industries['TOTAL'] = industries['EXITS AMOUNTS'].apply(number_to_string)
    industries['EXITS AMOUNTS'] = industries['EXITS AMOUNTS']*1000000 # To fix format

    ###### INVESTORS ######
    total_exits = total_exits.groupby('NAME').agg({'EXITS TOTAL (EUR M)':'sum','HQ COUNTRY':'first','LATITUDE':'first', 'LONGITUDE':'first'}).reset_index()
    total_exits = total_exits[total_exits['EXITS TOTAL (EUR M)'] != 0.00]
    total_exits = total_exits.dropna(subset=['LATITUDE', 'LONGITUDE'])
    total_exits.reset_index(drop = True)
    total_exits['EXITS AMOUNTS'] = total_exits['EXITS TOTAL (EUR M)'].apply(number_to_string)
    total_exits['EXITS TOTAL (EUR M)'] = total_exits['EXITS TOTAL (EUR M)']*1000000
    
    
    ### PREPROCESSSING PLOT HOME PAGE
        # Selecting required columns
    currently_unicorn = unicorn_df[['NAME', 'LATITUDE', 'LONGITUDE', 'COMPANY STATUS', 'INDUSTRIES']].copy()
    # Filtering operational unicorns
    operational_unicorns = currently_unicorn[currently_unicorn['COMPANY STATUS'] == 'operational']
    # Splitting industries and keeping the first one
    operational_unicorns['INDUSTRIES'] = operational_unicorns['INDUSTRIES'].astype(str).str.split(';')
    # Exploding operational unicorns by industries
    operational_unicorns = operational_unicorns.explode('INDUSTRIES')
    # Map industries to colors
    operational_unicorns['INDUSTRIES'] = operational_unicorns['INDUSTRIES'].map(industry_mapping)
    industry_colors = {
    "Technology & Software": "red",
    "Transportation & Logistics": "blue",
    "Energy & Engineering": "green",
    'Health & Wellness': "orange",
    "Robotics & Advanced Manufacturing": "purple",
    'Media & Entertainment': "yellow",
    "Finance & Real Estate": "cyan",
    "Consumer Goods & Services": "magenta",
    'Human Resources & Education': "lime",
    'Social & Lifestyle': "pink",
    }

    operational_unicorns["IndustryColor"] = operational_unicorns["INDUSTRIES"].map(industry_colors)
    # Drop rows with NaN in industries
    operational_unicorns = operational_unicorns.dropna(subset=['IndustryColor'])
    # Count the number of currently active unicorns
    active_unicorn_count = len(unicorn_df[unicorn_df['COMPANY STATUS'] == 'operational'])
    

    df_to_return = [unicorn_df, vcs, capital, unicorn_and_vc_per_year, 
                    time_needed, average_time, average_time_general, merged_data,
                    available_years, vc_genders, unicorn_gender, df_vc_extended, df_un_extended,
                    countries, continent_data_vc, continent_data_exit, founders_edu, founders,
                    industries, total_exits, industry_mapping, active_unicorn_count, operational_unicorns,
                    industry_colors]
    return df_to_return

def data_preprosessing_final_plot(unicorn_df2, vcs2, capital2, countries_df2):
    split = ['INVESTORS', 'EACH ROUND TYPE', 'EACH ROUND DATE']

    for col in split:

        unicorn_df2[col] = unicorn_df2[col].apply(lambda x: x.split(';') if isinstance(x, str) else (x if pd.notna(x) else []))

    unicorn_df2 = unicorn_df2.dropna(subset = ['INVESTORS'])

    # Transforming Each Round Date into DateTime
    unicorn_df2['EACH ROUND DATE'] = unicorn_df2['EACH ROUND DATE'].apply(lambda lst: [pd.to_datetime(date if '/' in date else f'jan/{date}', format='%b/%Y') for date in lst])

    # Specific Format
    unicorn_df2['EACH ROUND DATE'] = unicorn_df2['EACH ROUND DATE'].apply(lambda lst: [date.strftime('%m-%Y') if not pd.isnull(date) else '' for date in lst])

    # EACH ROUND INVESTORS
    unicorn_df2['EACH ROUND INVESTORS'] = unicorn_df2['EACH ROUND INVESTORS'].apply(lambda x: [investor_group.split('++') for investor_group in x.split(';')] if isinstance(x, str) else x)

    def map_dates_to_investors(row):
        # Initialize an empty dictionary for this row's mapping
        row_mapping = {}

        # Ensure that both date and investors are not NaN and are iterable
        dates = row['EACH ROUND DATE'] if isinstance(row['EACH ROUND DATE'], list) else []
        investors_lists = row['EACH ROUND INVESTORS'] if isinstance(row['EACH ROUND INVESTORS'], list) else []

        # Iterate over each date and corresponding investors list
        for date, investors in zip(dates, investors_lists):
            if isinstance(date, str) and investors:  # Ensure date is a string and investors is not empty
                if date in row_mapping:
                    if isinstance(row_mapping[date], list):
                        row_mapping[date].extend(investors if isinstance(investors, list) else [investors])
                    else:
                        row_mapping[date] = [row_mapping[date], investors] if isinstance(investors, list) else [row_mapping[date]] + [investors]
                else:
                    row_mapping[date] = investors if isinstance(investors, list) else [investors]

        return row_mapping

    unicorn_df2['DATE OF EACH ROUND INVESTORS'] = unicorn_df2.apply(map_dates_to_investors, axis=1)

    unicorn_df2 = unicorn_df2.explode('INVESTORS')

    def get_investment_date(investor, investment_dict):
        # Iterate through the dictionary to find the date associated with the investor
        for date, investors in investment_dict.items():
            if investor in investors:
                return date
        return None  # Return None if the investor is not found

    unicorn_df2['DATE OF INVESTMENT'] = unicorn_df2.apply(lambda row: get_investment_date(row['INVESTORS'], row['DATE OF EACH ROUND INVESTORS']), axis=1)

    unicorns = unicorn_df2[['NAME', 'HQ REGION', 'HQ COUNTRY', 'HQ CITY', 'LATITUDE', 'LONGITUDE', 'INVESTORS', 'DATE OF INVESTMENT']].copy().rename(columns={
        'NAME':'UNICORN',
        'HQ REGION':'UNICORN REGION',
        'HQ COUNTRY':'UNICORN COUNTRY',
        'HQ CITY':'UNICORN CITY',
        'LATITUDE': 'UNICORN LATITUDE',
        'LONGITUDE': 'UNICORN LONGITUDE'
    })

    investors = vcs2[['NAME', 'HQ REGION', 'HQ COUNTRY', 'HQ CITY', 'LATITUDE', 'LONGITUDE']].copy().rename(columns={
        'NAME': 'INVESTORS',
        'HQ REGION':'INVESTOR REGION',
        'HQ COUNTRY':'INVESTOR COUNTRY',
        'HQ CITY':'INVESTOR CITY',
        'LATITUDE': 'INVESTOR LATITUDE',
        'LONGITUDE': 'INVESTOR LONGITUDE'
    })

    unicorninvestors = pd.merge(unicorns, investors, how='inner', on='INVESTORS')

    unicorninvestors['YEAR OF INVESTMENT'] = pd.to_datetime(unicorninvestors['DATE OF INVESTMENT']).dt.year
    
    df_to_return = [unicorn_df2, vcs2, capital2, countries_df2, unicorninvestors]
    return df_to_return


def home_page_map(active_unicorn_count, operational_unicorns, industry_colors):
    # Create map title with active unicorn count
    map_title = f'Currently Active Unicors across the Globe: {active_unicorn_count}'

    # Define legend trace
    legend_trace = go.Scattergeo(
        lon=[],
        lat=[],
        mode='markers',
        marker=dict(
            size=1,  # Set marker size to 1 for a small point
            color=list(industry_colors.values()),  # Use industry colors for legend markers
            showscale=False,  # Do not show color scale in legend
            opacity=0.8
        ),
        showlegend=True,
        legendgroup="unicorns",  # Assign a legend group
        name="Industry"  # Name for the legend
    )

    # Create a map using Plotly
    fig = go.Figure(go.Scattergeo(
        lat=operational_unicorns["LATITUDE"].astype(float),  # Ensure latitudes are float type
        lon=operational_unicorns["LONGITUDE"].astype(float),  # Ensure longitudes are float type
        mode="markers",
        hoverinfo="text",
        hovertext=operational_unicorns["NAME"] + "<br>Industry: " + operational_unicorns["INDUSTRIES"],
        marker=dict(
            size=10,
            color='red',
            opacity=0.8
        )
    ))

    # Add legend trace to figure
    fig.add_trace(legend_trace)

    # Update layout settings for map
    fig.update_layout(
        #title=map_title,  # Set map title with active unicorn count
        geo=dict(
            scope='world',
            showland=True,
            landcolor='rgb(0, 100, 0)',
            showcountries=True,
            countrycolor='rgb(0, 0, 0)',
            showocean=True,
            oceancolor='rgb(204, 204, 255)',
            showcoastlines=True,
            coastlinecolor='rgb(0,0,0)',
            showframe=True,
            framecolor='rgb(0,0,0)',
            bgcolor='rgb(0, 0, 0)',
            projection=dict(
            type='natural earth'
            )
        ),
        width=1200,
        height=800,
        paper_bgcolor='black',
        plot_bgcolor='black',
        title=map_title
    
    )
    
    fig.update_layout(
                    title = {
                            'text': map_title,  # Title text
                            'y':0.95,  # Position of the title (0.0 to 1.0, top to bottom)
                            'x':0.5,  # Position of the title (0.0 to 1.0, left to right, 0.5 is center)
                            'xanchor': 'center',  # Ensures the title is centered at the x position
                            'yanchor': 'top',  # Anchors the title to the top of the plot
                            'font': {
                                'family': "Arial",  # Font family
                                'size': 24,  # Font size
                                'color': "orange"  # Font color
                    }
            }
    )
    return fig


# Define the plots
def plot_unicorn_vc_trends(unicorn_and_vc_per_year, average_time, average_time_general, industry_colors):
    visible_industries = []
    fig1 = go.Figure()

    # Plot the general trend as a fixed line
    fig1.add_trace(go.Scatter(x=average_time_general['LAUNCH YEAR'], y=average_time_general['time_needed'],
                         mode='lines', name='General Trend'))
    # Add dropdown menu
    ## fig1.update_layout(
    ##     updatemenus=[
    ##         dict(
    ##             buttons=[{'label': 'All Industries', 'method': 'update', 'args': [{'visible': True}]},
    ##                      {'label': 'None', 'method': 'update', 'args': [{'visible': False}]}],
    ##             direction='down',
    ##             showactive=True,
    ##             x=1.02,  # Adjusted x position
    ##             xanchor='left',
    ##             y=0.65,
    ##             yanchor='top'
    ##         )
    ##     ]
    ## )

    # Function to update visible traces based on dropdown selection
    def update_visible_traces(trace):
        inds = [i for i, val in enumerate(fig1.data) if val.visible == 'legendonly']
        if inds:
            for i in inds:
                fig1.data[i].visible = True if trace.name == fig1.data[i].name else False

    # Add traces for each industry
    for industry in average_time['INDUSTRIES'].unique():
        industry_data = average_time[average_time['INDUSTRIES'] == industry]
        visible = True if industry == 'Selected Industry' else 'legendonly'
        fig1.add_trace(go.Scatter(
            x=industry_data['LAUNCH YEAR'],
            y=industry_data['average_time_needed_per_industry_per_year'],
            mode='lines',
            name=industry,
            visible=visible,
            line=dict(color=industry_colors.get(industry, '#000'))  # Default to black if no mapping
        ))
        visible_industries.append(industry)

    # Update layout
    fig1.update_layout(
        title={
        'text': "Evolution of the Mean Time needed for Startups to Become Unicorns",  # Title text
        'y':0.95,  # Position of the title (0.0 to 1.0, top to bottom)
        'x':0.5,  # Position of the title (0.0 to 1.0, left to right, 0.5 is center)
        'xanchor': 'center',  # Ensures the title is centered at the x position
        'yanchor': 'top',  # Anchors the title to the top of the plot
        'font': {
            'family': "Arial",  # Font family
            'size': 24,  # Font size
            'color': "orange"  # Font color
        }},
        xaxis_title='Launch Year',
        yaxis_title='Average Time Needed',
        hovermode='x unified'
    )
    
    fig1.update_layout({'plot_bgcolor': 'black', 
                        'paper_bgcolor': 'black'},
                         width=1200,  # Width of the plot
                         height=800 # Background color    
                         )


    return fig1

def plot_time_to_unicorn(merged_data, industry_colors):
    fig2 = px.bar(
        merged_data,
        x="CUMULATIVE UNICORN COUNT",
        y="INDUSTRIES",
        color="INDUSTRIES",
        animation_frame="LAUNCH YEAR",
        orientation='h',  # Makes the bars horizontal
        range_x=[0, merged_data['CUMULATIVE UNICORN COUNT'].max()],
        category_orders={"LAUNCH YEAR": sorted(merged_data['LAUNCH YEAR'].unique())},
        color_discrete_map=industry_colors  # Apply the industry color mapping
    )


    fig2.update_layout(
        yaxis={
            'categoryorder': 'total ascending'
        },
        title = {
                            'text': 'The Growth of Industries Across Time',  # Title text
                            'y':0.97,  # Position of the title (0.0 to 1.0, top to bottom)
                            'x':0.5,  # Position of the title (0.0 to 1.0, left to right, 0.5 is center)
                            'xanchor': 'center',  # Ensures the title is centered at the x position
                            'yanchor': 'top',  # Anchors the title to the top of the plot
                            'font': {
                                'family': "Arial",  # Font family
                                'size': 24,  # Font size
                                'color': "orange"  # Font color
                    }
            },
        xaxis_title="Cumulative Unicorn Count",
        xaxis={
            'showgrid': True,  # Enables grid lines on the x-axis
            'gridcolor': 'grey'  # Sets grid line color to grey
        },
        yaxis_title="Industries",
        plot_bgcolor='black',  # Background color of the plotting area
        paper_bgcolor='black',  # Background color of the surrounding paper
        width=1200,  # Width of the plot
        height=700  # Height of the plot
    )

    return fig2



def plot_gender_analysis(vc_genders, unicorn_gender, available_years):

# Create frames for each year
    frames = []
    for year in available_years:
        # Filter data for the current year
        vc_data_year = vc_genders[vc_genders['LAUNCH YEAR'].dt.year == year]
        unicorn_data_year = unicorn_gender[unicorn_gender['LAUNCH YEAR'].dt.year == year]

        # Define traces for VC data
        vc_trace = go.Scattergeo(
            lon=vc_data_year['longitude'],
            lat=vc_data_year['latitude'],
            text=vc_data_year['LAUNCH YEAR'],
            mode='markers',
            marker=dict(
                size=vc_data_year['cumulative_women_count_in_vc'],  # Marker size based on cumulative women count for VC
                color='blue',  # Marker color for VC
                line=dict(width=0.5, color='rgb(40,40,40)'),
                sizemode='area',
                sizemin=5  # Set minimum size for markers
            ),
            name='VC'
        )

        # Define traces for Unicorn data
        unicorn_trace = go.Scattergeo(
            lon=unicorn_data_year['longitude'],
            lat=unicorn_data_year['latitude'],
            text=unicorn_data_year['LAUNCH YEAR'],
            mode='markers',
            marker=dict(
                size=unicorn_data_year['cumulative_women_count_in_unicorn'],  # Marker size based on cumulative women count for Unicorn
                color='red',  # Marker color for Unicorn
                line=dict(width=0.5, color='rgb(40,40,40)'),
                sizemode='area',
                sizemin=5  # Set minimum size for markers
            ),
            name='Unicorn'
        )

        # Create a frame for the current year
        frame = {"data": [vc_trace, unicorn_trace], "name": str(year)}
        frames.append(frame)

    # Create slider steps
    slider_steps = []
    for i, year in enumerate(available_years):
        step = {"args": [
            [str(year)],  # Frame name
            {"frame": {"duration": 300, "redraw": True}, "mode": "immediate", "transition": {"duration": 300}}
        ],
                "label": str(year),
                "method": "animate"}
        slider_steps.append(step)

    # Create figure
    fig_3 = go.Figure(
        data=[frames[0]["data"][0], frames[0]["data"][1]],
        layout=go.Layout(
        
            #title='Cumulative Number of Women in VC and Unicorn',
        geo=dict(
            scope='world',
            showland=True,
            landcolor='rgb(0, 100, 0)',
            showcountries=True,
            countrycolor='rgb(0, 0, 0)',
            showocean=True,
            oceancolor='rgb(204, 204, 255)',
            showcoastlines=True,
            coastlinecolor='rgb(0,0,0)',
            showframe=True,
            framecolor='rgb(0,0,0)',
            bgcolor='rgb(0, 0, 0)',
            projection=dict(
            type='natural earth'
            )
        ),
            hoverlabel=dict(
                bgcolor="white",  # Set hover label background color to white
                font_size=12,  # Set hover label font size
                font_family="Rockwell"  # Set hover label font family
            ),
            sliders=[{
                "active": 0,
                "steps": slider_steps,
                "pad": {"t": 50},
                "len": 0.9,
                "x": 0.1,
                "xanchor": "left",
                "y": 0,
                "yanchor": "top",
                "currentvalue": {"font": {"size": 20}, "prefix": "Year:", "visible": True, "xanchor": "right"}
            }],
        ),
        frames=[frames[i] for i in range(len(frames))]
    )
    
    fig_3.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
    )

    fig_3.update_layout(
    paper_bgcolor='rgb(0, 0, 0)',  # Sets the surrounding area to black
    plot_bgcolor='rgb(0, 0, 0)',  # Sets the plot area to black
    width=1200,  # Width of the plot
    height=700,
    title = {
                    'text': 'A Global Timeline  of the Increase in Womens\'s Role',  # Title text
                    'y':0.97,  # Position of the title (0.0 to 1.0, top to bottom)
                    'x':0.5,  # Position of the title (0.0 to 1.0, left to right, 0.5 is center)
                    'xanchor': 'center',  # Ensures the title is centered at the x position
                    'yanchor': 'top',  # Anchors the title to the top of the plot
                    'font': {
                        'family': "Arial",  # Font family
                        'size': 24,  # Font size
                        'color': "orange"  # Font color
                    }
            }
    )
    return fig_3

def plot_vc_vs_unicorn_map(df_vc_extended, df_un_extended):
    trace_vc = go.Scattergeo(
        lon=df_vc_extended['LONGITUDE'],
        lat=df_vc_extended['LATITUDE'],
        mode='markers',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.8,
            symbol='circle'
        ),
        name='Venture Capital',
        hoverinfo='text',
        text=df_vc_extended.apply(lambda row: f"Name: {row['NAME']}<br>Launch Year: {row['LAUNCH YEAR']}<br>HQ City: {row['HQ CITY']}", axis=1)
    )

    # Modifica della traccia per le unicorn
    trace_unicorn = go.Scattergeo(
        lon=df_un_extended['LONGITUDE'],
        lat=df_un_extended['LATITUDE'],
        mode='markers',
        marker=dict(
            size=8,
            color='red',
            opacity=0.8,
            symbol='circle'
        ),
        name='Unicorn',
        hoverinfo='text',
        text=df_un_extended.apply(lambda row: f"Name: {row['NAME']}<br>Launch Year: {row['LAUNCH YEAR']}<br>HQ City: {row['HQ CITY']}", axis=1)
    )


    # Creazione del layout della mappa
    layout = go.Layout(
        title={
        'text': "Venture Capital and Unicorn Map",  # Title text
        'y':0.95,  # Position of the title (0.0 to 1.0, top to bottom)
        'x':0.5,  # Position of the title (0.0 to 1.0, left to right, 0.5 is center)
        'xanchor': 'center',  # Ensures the title is centered at the x position
        'yanchor': 'top',  # Anchors the title to the top of the plot
        'font': {
            'family': "Arial",  # Font family
            'size': 24,  # Font size
            'color': "orange"  # Font color
        }},
        paper_bgcolor='rgb(0, 0, 0)',  # Sets the surrounding area to black
        plot_bgcolor='rgb(0, 0, 0)', 
        geo=dict(
            scope='world',
            showland=True,
            landcolor='rgb(0, 100, 0)',
            showcountries=True,
            countrycolor='rgb(0, 0, 0)',
            showocean=True,
            oceancolor='rgb(204, 204, 255)',
            showcoastlines=True,
            coastlinecolor='rgb(0,0,0)',
            showframe=True,
            framecolor='rgb(0,0,0)',
            bgcolor='rgb(0, 0, 0)',
            projection=dict(
                type='natural earth'
            )
        ),
        updatemenus=[
            {
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }
        ],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Year:',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [{'args': [[str(year)], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}],
                       'label': str(year),
                       'method': 'animate'} for year in sorted(df_vc_extended['LAUNCH YEAR'].unique())]
        }]
    )
    # Creazione dei frame animati con aggiornamento del campo hoverinfo
    frames = [
        go.Frame(
            data=[
                go.Scattergeo(
                    lon=df_vc_extended[df_vc_extended['LAUNCH YEAR'] == year]['LONGITUDE'],
                    lat=df_vc_extended[df_vc_extended['LAUNCH YEAR'] == year]['LATITUDE'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='blue',
                        opacity=0.8,
                        symbol='circle'
                    ),
                    name='Venture Capital',
                    hoverinfo='text',
                    text=df_vc_extended[df_vc_extended['LAUNCH YEAR'] == year].apply(lambda row: f"Name: {row['NAME']}<br>Launch Year: {row['LAUNCH YEAR']}<br>HQ City: {row['HQ CITY']}", axis=1)
                ),
                go.Scattergeo(
                    lon=df_un_extended[df_un_extended['LAUNCH YEAR'] == year]['LONGITUDE'],
                    lat=df_un_extended[df_un_extended['LAUNCH YEAR'] == year]['LATITUDE'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        opacity=0.8,
                        symbol='circle'
                    ),
                    name='Unicorn',
                    hoverinfo='text',
                    text=df_un_extended[df_un_extended['LAUNCH YEAR'] == year].apply(lambda row: f"Name: {row['NAME']}<br>Launch Year: {row['LAUNCH YEAR']}<br>HQ City: {row['HQ CITY']}", axis=1)
                )
            ],
            name=str(year)
        )
        for year in sorted(df_vc_extended['LAUNCH YEAR'].unique())
    ]
    # Aggiunta dei frame alla figura
    fig = go.Figure(data=[trace_vc, trace_unicorn], layout=layout, frames=frames)
    fig.update_layout(
    width=1200,  # Width of the plot
    height=700)

    return fig

def create_continent_VC_plot(continent_data_vc):
    fig = go.Figure(data=[
        go.Bar(
            x=continent_data_vc['Continent'],
            y=continent_data_vc['TOTAL VC FUNDING'],
            text=continent_data_vc['TOTAL FUNDING'],
            textposition='outside',
        )
    ])
    fig.update_layout(title = {
                            'text': 'Total VC Funding by Continent',  # Title text
                            'y':0.97,  # Position of the title (0.0 to 1.0, top to bottom)
                            'x':0.5,  # Position of the title (0.0 to 1.0, left to right, 0.5 is center)
                            'xanchor': 'center',  # Ensures the title is centered at the x position
                            'yanchor': 'top',  # Anchors the title to the top of the plot
                            'font': {
                                'family': "Arial",  # Font family
                                'size': 24,  # Font size
                                'color': "orange"  # Font color
                            }
                     },
                      xaxis_title="Continent",
                      yaxis_title="Total VC Funding",
                      plot_bgcolor='black',  # Background color of the plotting area
                      paper_bgcolor='black',  # Background color of the surrounding paper
                        width=1200,  # Width of the plot
                        height=700  # Height of the plot
        )
    return fig

def create_continent_exit_plot(continent_data_exit):
    fig = go.Figure(data=[
        go.Bar(
            x=continent_data_exit['Continent'],
            y=continent_data_exit['TOTAL EXIT AMOUNTS'],
            text=continent_data_exit['AMOUNT OF EXITS (2000)'],
            textposition='outside',
        )
    ])
    fig.update_layout(
                              title = {
                                    'text': 'Total Exit Amounts by Continent',  # Title text
                                    'y':0.97,  # Position of the title (0.0 to 1.0, top to bottom)
                                    'x':0.5,  # Position of the title (0.0 to 1.0, left to right, 0.5 is center)
                                    'xanchor': 'center',  # Ensures the title is centered at the x position
                                    'yanchor': 'top',  # Anchors the title to the top of the plot
                                    'font': {
                                    'family': "Arial",  # Font family
                                    'size': 24,  # Font size
                                    'color': "orange"  # Font color
                                             }
                                      },
                      xaxis_title="Continent", 
                      yaxis_title="Total Exit Amounts",
                    plot_bgcolor='black',  # Background color of the plotting area
                    paper_bgcolor='black',  # Background color of the surrounding paper
                    width=1200,  # Width of the plot
                    height=700)  # Height of the plot)
    return fig

def best_unis_investors(founders_edu):
    unis = founders_edu.explode('FOUNDERS UNIVERSITIES')
    uni_series = unis.value_counts('FOUNDERS UNIVERSITIES')
    uni_df = uni_series.reset_index()  
    uni_df.columns = ['UNIVERSITY', 'COUNT']
    
    # Uni most investors went to
    most_uni = uni_df.nlargest(10, 'COUNT')
    fig = px.bar(
        most_uni,
        x='UNIVERSITY',
        y='COUNT',
        title='Most attended Universities by Founders',
        labels={'COUNT': 'Number of Founders', 'UNIVERSITY': 'University'},
    )
    # Customize the layout
    fig.update_layout(
        xaxis_title='University',
        yaxis_title='Number of Founders',
        xaxis_tickangle=45,  # Rotate the labels for better visibility if needed
        paper_bgcolor='black',  # Background color of the surrounding paper
        plot_bgcolor='black',  # Background color of the plotting area
        width=1200,  # Width of the plot
        height=700,  # Height of the plot
        title = {
                    'text': 'Universities with The Most Alumni Founders',  # Title text
                    'y':0.97,  # Position of the title (0.0 to 1.0, top to bottom)
                    'x':0.5,  # Position of the title (0.0 to 1.0, left to right, 0.5 is center)
                    'xanchor': 'center',  # Ensures the title is centered at the x position
                    'yanchor': 'top',  # Anchors the title to the top of the plot
                    'font': {
                        'family': "Arial",  # Font family
                        'size': 24,  # Font size
                        'color': "orange"  # Font color
                        }
            }
    )

    return fig

def what_funders_studied(founders_edu):
    backgrounds = founders_edu.explode('FOUNDERS BACKGROUNDS')
    backgrounds_series = backgrounds.value_counts('FOUNDERS BACKGROUNDS')
    backgrounds_df = backgrounds_series.reset_index()

    fig = px.bar(
        backgrounds_df,
        x='FOUNDERS BACKGROUNDS',
        y='count',
        title='What Founders Studied',
        labels={'COUNT': 'Number of Founders', 'FOUNDERS BACKGROUNDS': 'Educational Background'},
    )

    # Customize the layout
    fig.update_layout(
        xaxis_title='Educational Background',
        yaxis_title='Number of Founders',
        xaxis_tickangle=45,  # Rotate the labels for better visibility if needed
        paper_bgcolor='black',  # Background color of the surrounding paper
        plot_bgcolor='black',  # Background color of the plotting area
        width=1200,  # Width of the plot
        height=700, # Height of the plot
        title = {
                    'text': 'Most Popular Fields of Study for Founders',  # Title text
                    'y':0.97,  # Position of the title (0.0 to 1.0, top to bottom)
                    'x':0.5,  # Position of the title (0.0 to 1.0, left to right, 0.5 is center)
                    'xanchor': 'center',  # Ensures the title is centered at the x position
                    'yanchor': 'top',  # Anchors the title to the top of the plot
                    'font': {
                        'family': "Arial",  # Font family
                        'size': 24,  # Font size
                        'color': "orange"  # Font color
                            }
            }
    )

    # Adding more customizations: adjust the text on the bars
    return fig

def first_company(founders):
    first = founders.explode('IS FOUNDERS FIRST COMPANY')
    first_counts = first.value_counts('IS FOUNDERS FIRST COMPANY')
    first_df = first_counts.reset_index()
    fig = px.pie(
        first_df, 
        values='count', 
        names='IS FOUNDERS FIRST COMPANY', 
        title='Is it the Founders First Company?',
        hole=0.4
    )

    fig.update_traces(
        textinfo='percent+label',
        hoverinfo='label+value',  # Show more details on hover
        pull=[0.1 if x == 'yes' else 0 for x in first_df['IS FOUNDERS FIRST COMPANY']] 
    )

    fig.update_layout(
        legend_title_text='First Company',
        paper_bgcolor='black',  # Background color of the surrounding paper
        plot_bgcolor='black',  # Background color of the plotting area
        title = {
                    'text': 'Percentage of First-Time Founders',  # Title text
                    'y':0.97,  # Position of the title (0.0 to 1.0, top to bottom)
                    'x':0.5,  # Position of the title (0.0 to 1.0, left to right, 0.5 is center)
                    'xanchor': 'center',  # Ensures the title is centered at the x position
                    'yanchor': 'top',  # Anchors the title to the top of the plot
                    'font': {
                        'family': "Arial",  # Font family
                        'size': 24,  # Font size
                        'color': "orange"  # Font color
                        }
            }
    )
    return fig

def for_industries(industries):
    fig = go.Figure(data=[
    go.Bar(
        x=industries['INDUSTRIES'], 
        y=industries['EXITS AMOUNTS'],  
        text=industries['TOTAL'],  
        textposition='outside' 
    )
    ])

    # Update layout for better visualization
    fig.update_layout(
        title = {
                    'text': 'Exits Amounts by Industries',  # Title text
                    'y':0.97,  # Position of the title (0.0 to 1.0, top to bottom)
                    'x':0.5,  # Position of the title (0.0 to 1.0, left to right, 0.5 is center)
                    'xanchor': 'center',  # Ensures the title is centered at the x position
                    'yanchor': 'top',  # Anchors the title to the top of the plot
                    'font': {
                    'family': "Arial",  # Font family
                    'size': 24,  # Font size
                    'color': "orange"  # Font color
                        }
            },
        xaxis_title='Industries',
        yaxis_title='Exits Amounts',
        xaxis=dict(
            tickfont=dict(color='white'),  
            titlefont=dict(color='white'), 
            linecolor='white', 
            tickcolor='white'
        ),
        yaxis=dict(
            tickfont=dict(color='white'),  
            titlefont=dict(color='white'), 
            linecolor='white',
            tickcolor='white' 
        ),
        plot_bgcolor='black',
        paper_bgcolor='black',
        width=1200,  # Width of the plot
        height=700  # Height of the plot
    )
    return fig

def for_investors(total_exits):
    largest_vcs = total_exits.nlargest(10, 'EXITS TOTAL (EUR M)')
    fig = go.Figure(data=[
        go.Bar(
            x=largest_vcs['NAME'], 
            y=largest_vcs['EXITS TOTAL (EUR M)'],  
            text=largest_vcs['EXITS AMOUNTS'],  
            textposition='outside'
        )
    ])

    fig.update_traces(
        hoverinfo='name+y+text', 
        hovertemplate="<b>%{x}</b><br><br>Exits Total: %{y}B EUR<br>HQ Country: %{customdata[0]}<extra></extra>",
        customdata=largest_vcs[['HQ COUNTRY']]
    )

    # Update layout for better visualization
    fig.update_layout(
        title = {
                    'text': 'Investors with Most Exit Totals',  # Title text
                    'y':0.97,  # Position of the title (0.0 to 1.0, top to bottom)
                    'x':0.5,  # Position of the title (0.0 to 1.0, left to right, 0.5 is center)
                    'xanchor': 'center',  # Ensures the title is centered at the x position
                    'yanchor': 'top',  # Anchors the title to the top of the plot
                    'font': {
                    'family': "Arial",  # Font family
                    'size': 24,  # Font size
                    'color': "orange"  # Font color
                        }
            },
        xaxis_title='Investors',
        yaxis_title='Exits Amounts',
        xaxis=dict(
            tickfont=dict(color='white'),  
            titlefont=dict(color='white'), 
            linecolor='white', 
            tickcolor='white'
        ),
        yaxis=dict(
            tickfont=dict(color='white'),  
            titlefont=dict(color='white'), 
            linecolor='white',
            tickcolor='white' 
        ),
        plot_bgcolor='black',
        paper_bgcolor='black',
        width=1200,  # Width of the plot
        height=700  # Height of the plot
    )
    return fig

def cluster_bubble_chart(total_exits, ):
    coords = total_exits[['LATITUDE', 'LONGITUDE']].apply(np.radians)

    # Configure DBSCAN
    kms_per_radian = 6371.0088  # Approximate radius of the earth in kilometers; use a suitable radius in km
    epsilon = 20 / kms_per_radian  # 1 km search radius
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine')
    cluster_labels = db.fit_predict(coords)

    # Assign cluster labels back to the DataFrame
    total_exits['cluster'] = cluster_labels

    # Aggregate exit amounts by cluster
    clustered_data = total_exits.groupby('cluster').agg({
        'EXITS TOTAL (EUR M)': 'sum',
        'LATITUDE': 'mean',  
        'LONGITUDE': 'mean',
        'NAME': lambda x: ', '.join(x)
    }).reset_index()

    # New function needed to convert because of conversion of scale for 'EXITS TOTAL (EUR M)'

    def new_number_to_string(number):
        abs_number = abs(number)

        if abs_number >= 1_000_000_000_000:  # Trillions
            formatted_number = f"{number / 1_000_000_000_000:.0f}T"
        elif abs_number >= 1_000_000_000:  # Billions
            formatted_number = f"{number / 1_000_000_000:.0f}B"
        elif abs_number >= 1_000_000:  # Millions
            formatted_number = f"{number / 1_000_000:.0f}M"
        elif abs_number >= 1_000:  # Thousands
            formatted_number = f"{number / 1_000:.0f}K"
        else:  # Less than 1000
            formatted_number = f"{number:.0f}"

        return f"${formatted_number}"

    # Use your conversion function here if needed
    clustered_data['EXITS AMOUNTS'] = clustered_data['EXITS TOTAL (EUR M)'].apply(new_number_to_string)

    def format_names(names):
        names_list = names.split(', ')
        if len(names_list) > 5:  # Adjust the number as needed
            return '<br>'.join(names_list[:5]) + f'<br>and {len(names_list) - 5} more...'
        else:
            return '<br>'.join(names_list)

    clustered_data['NAME'] = clustered_data['NAME'].apply(format_names)

    # For scaling
    color_scale = "Bluered"  

    # Create the plot
    fig = px.scatter_geo(
        clustered_data,
        lat='LATITUDE',
        lon='LONGITUDE',
        size='EXITS TOTAL (EUR M)',  
        color='EXITS TOTAL (EUR M)', 
        color_continuous_scale=color_scale,  
        hover_name='NAME',
        hover_data={
            'EXITS TOTAL (EUR M)': False,
            'EXITS AMOUNTS': True,
            'LATITUDE': False,
            'LONGITUDE':False  
        },
        projection="natural earth",
        size_max=50  
    )

    # Show color scale bar
    fig.update_layout(coloraxis_colorbar=dict(
        title="Exit Amounts",
        tickvals=[clustered_data['EXITS TOTAL (EUR M)'].min(), clustered_data['EXITS TOTAL (EUR M)'].max()],
        ticktext=['Low', 'High']),
        geo=dict(
            scope='world',
            showland=True,
            landcolor='rgb(0, 100, 0)',
            showcountries=True,
            countrycolor='rgb(0, 0, 0)',
            showocean=True,
            oceancolor='rgb(204, 204, 255)',
            showcoastlines=True,
            coastlinecolor='rgb(0,0,0)',
            showframe=True,
            framecolor='rgb(0,0,0)',
            bgcolor='rgb(0, 0, 0)',
            projection=dict(
            type='natural earth'
            )),
        title = {
                    'text': 'Investor Profits Clustered by Region',  # Title text
                    'y':0.97,  # Position of the title (0.0 to 1.0, top to bottom)
                    'x':0.5,  # Position of the title (0.0 to 1.0, left to right, 0.5 is center)
                    'xanchor': 'center',  # Ensures the title is centered at the x position
                    'yanchor': 'top',  # Anchors the title to the top of the plot
                    'font': {
                        'family': "Arial",  # Font family
                        'size': 24,  # Font size
                        'color': "orange"  # Font color
            }
        },
        width=1200,  # Width of the plot
        height=700,  # Height of the plot
        paper_bgcolor='black',  # Background color of the surrounding paper
        plot_bgcolor='black'  # Background color of the plotting area
    )
    return fig

def introduction_plot(unicorn_df, industry_mapping):
    currently_unicorn= unicorn_df[['NAME', 'LATITUDE', 'LONGITUDE', 'COMPANY STATUS', 'INDUSTRIES', "WEBSITE"]]
    operational_unicorns = currently_unicorn[currently_unicorn['COMPANY STATUS'] == 'operational']

    currently_unicorn['INDUSTRIES']= currently_unicorn['INDUSTRIES'].astype(str).str.split(';').tolist()
    currently_unicorn = currently_unicorn['INDUSTRIES'].apply(lambda x: x[0])
    operational_unicorns= operational_unicorns.explode('INDUSTRIES')
    operational_unicorns['INDUSTRIES']= operational_unicorns['INDUSTRIES'].map(industry_mapping)
    operational_unicorns= operational_unicorns.dropna()


    unique_industries = operational_unicorns['INDUSTRIES'].unique()

    industry_colors = {
        "Technology & Software": "red",
        "Transportation & Logistics": "blue",
        "Energy & Engineering": "green",
        'Health & Wellness': "orange",
        "Robotics & Advanced Manufacturing": "purple",
        'Media & Entertainment': "yellow",
        "Finance & Real Estate": "cyan",
        "Consumer Goods & Services": "magenta",
        'Human Resources & Education': "lime",
        'Social & Lifestyle': "pink",
    }

    operational_unicorns['IndustryColor'] = operational_unicorns['INDUSTRIES'].map(industry_colors)
    fig = go.Figure(go.Scattermapbox(
    lat=operational_unicorns["LATITUDE"],
    lon=operational_unicorns["LONGITUDE"],
    mode="markers",
    hoverinfo="text",
    hovertext=operational_unicorns["NAME"] + "<br>Industry: " + operational_unicorns["INDUSTRIES"],
    marker=dict(
        size=10,
        color=operational_unicorns["IndustryColor"],
        opacity=0.8
    )
    ))

    # Update layout settings for map
    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"r":0,"t":0,"l":0,"b":0},
        mapbox=dict(
            zoom=2
        )
    )
    return fig

def evolution_vcs_uni(unicorn_df, vcs):
    unicorns_per_year = unicorn_df.groupby('LAUNCH YEAR')['NAME'].count()
    unicorns_per_year= unicorns_per_year.to_frame()
    unicorns_per_year= unicorns_per_year.reset_index()
    unicorns_per_year.rename(columns={'NAME': 'NUMBER OF UNICORNS FOUNDED'}, inplace=True)
    unicorns_per_year['Cumulative Unicorn FOUNDED'] = unicorns_per_year['NUMBER OF UNICORNS FOUNDED'].cumsum()
    unicorns_per_year['LAUNCH YEAR']= pd.to_datetime(unicorns_per_year['LAUNCH YEAR'], format='%Y')



    vc_per_year= vcs.groupby('LAUNCH YEAR')['NAME'].count()
    vc_per_year= vc_per_year.to_frame()
    vc_per_year= vc_per_year.reset_index()
    vc_per_year.rename(columns={'NAME': 'NUMBER OF VC FOUNDED'}, inplace=True)
    vc_per_year = vc_per_year[vc_per_year['LAUNCH YEAR'] >= 1980]
    vc_per_year['Cumulative VC Founded'] = vc_per_year['NUMBER OF VC FOUNDED'].cumsum()
    vc_per_year['LAUNCH YEAR']= pd.to_datetime(vc_per_year['LAUNCH YEAR'], format='%Y')

    unicorn_and_vc_per_year= pd.merge(vc_per_year, unicorns_per_year, on='LAUNCH YEAR', how='outer')
    unicorn_and_vc_per_year['LAUNCH YEAR'] = pd.to_datetime(unicorn_and_vc_per_year['LAUNCH YEAR']).dt.year
    unicorn_and_vc_per_year.fillna(method='ffill', inplace=True)
    unicorn_and_vc_per_year= unicorn_and_vc_per_year.fillna(0)

    fig = px.line(unicorn_and_vc_per_year, x='LAUNCH YEAR', y=['Cumulative VC Founded', 'Cumulative Unicorn FOUNDED'],
                  labels={'LAUNCH YEAR': 'Years', 'value': 'Number of Players Founded', 'variable': 'Type'},
                  title="Trend of Players Founded Over Years",
                  color_discrete_map={'Cumulative VC Founded': 'blue', 'Cumulative Unicorn FOUNDED': 'red'})

    fig.update_layout({
        'plot_bgcolor': 'black', 
        'paper_bgcolor': 'black', 
        'legend_title_text': '', 
        'legend_x': 0.01, 
        'legend_y': 0.99, 
        'legend_bgcolor': 'rgba(255, 255, 255, 0.5)', 
        'xaxis': {'showgrid': True, 'gridwidth': 1, 'gridcolor': 'rgba(200, 200, 200, 0.5)'}, 
        'yaxis': {'showgrid': True, 'gridwidth': 1, 'gridcolor': 'rgba(200, 200, 200, 0.5)'}
    },
        title={
        'text': "Trend of Players Founded Over the Years",  # Title text
        'y':0.95,  # Position of the title (0.0 to 1.0, top to bottom)
        'x':0.5,  # Position of the title (0.0 to 1.0, left to right, 0.5 is center)
        'xanchor': 'center',  # Ensures the title is centered at the x position
        'yanchor': 'top',  # Anchors the title to the top of the plot
        'font': {
            'family': "Arial",  # Font family
            'size': 24,  # Font size
            'color': "orange"  # Font color
        }},
    height=600)

    fig.update_traces(line=dict(width=2.5), selector=dict(type='scatter', mode='lines'))

    return fig




    

def main():
    page_bg_image = '''
        <style>
        [data-testid="stAppViewContainer"] {
        background-image: url("https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjYxMy10YXVzLTEyLXVyYmFuZ3JhZGllbnRiYWNrZ3JvdW5kXzEuanBn.jpg") !important;
        background-size: cover;    
        }

        [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);
        }

        [data-testid="stToolbar"] {
        right: 2rem;
        }
        </style>
    '''



    st.markdown(page_bg_image, unsafe_allow_html=True)
    #st.title('The Evolution of Unicorn Ecosystems')
    page = st.sidebar.selectbox("Choose a topic", ["Home Page", "The Market Evolution", "Investments","Investments Flow" ,"Backgrounds", "The Evolution of Womens' Roles", "About Us"])

    unicorn_df, vcs, capital, countries_df, unicorn_df2, vcs2, capital2, countries_df2, = load_data()
    (unicorn_df, vcs, capital, unicorn_and_vc_per_year,
        time_needed, average_time, average_time_general, merged_data,
        available_years, vc_genders, unicorn_gender, df_vc_extended, df_un_extended,
        countries, continent_data_vc, continent_data_exit, founders_edu, founders,
        industries, total_exits, industry_mapping,
        active_unicorn_count, operational_unicorns, industry_colors) = preprocess_data(unicorn_df, vcs, capital, countries_df)
    (unicorn_df2, vcs2, capital2, countries_df2, unicorninvestors) = data_preprosessing_final_plot(unicorn_df2, vcs2, capital2, countries_df2)
    


    if page == "Home Page":
        st.markdown("""
            <div style='text-align: center'>
                <h1>The Evolution of Unicorn Ecosystems</h1>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <style>
            /* Adjusts the padding and maximum width of the main container */
            .appview-container .main .block-container {
                max-width: 100%;  /* Full width */
                padding-top: 1rem;  /* Top padding */
                padding-right: 3rem;  /* Right padding */
                padding-left: 2rem;  /* Left padding */
                padding-bottom: 1rem;  /* Bottom padding */
            }
            /* Optionally hides footer and file uploader button if not needed */
            .uploadedFile { display: none; }
            footer { visibility: hidden; }
        </style>
        """, unsafe_allow_html=True)
        
        css = """
        <style>
            .blur-box {
                background-color: rgba(0, 0, 0, 0.5);  /* White background with 50% opacity */
                backdrop-filter: blur(8px);  /* Blur effect */
                border-radius: 10px;  /* Rounded corners */
                padding: 20px;  /* Padding inside the box */
                margin: 10px;  /* Margin around the box */
                color: white;  /* Text color */
                font-size: 16px;  /* Font size */
            }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
        st.markdown("""
            <div class='blur-box'>
                Welcome to the <span style="font-weight: bold; color: #FFA500">Unicorns</span> <span style="font-weight: bold; color: #FFA500">Ecosystem</span> <span style="font-weight: bold; color: #FFA500">website</span>! 
                Here, you’ll be able to answer all the questions you may have on Unicorns and the players that they interact with.
                First of all, what is a Unicorn, you may ask? Well, a Unicorn is defined as a start-up with a valuation of over <span style="font-weight: bold; color: #FFA500">one</span> <span style="font-weight: bold; color: #FFA500">BILLION</span> <span style="font-weight: bold; color: #FFA500">dollars</span>! On this website,
                you’ll be able to learn about the history of unicorns, see how they evolved over time, and see what their current environment looks like.
                You’ll also be able to learn about their supporting cast, that is, the companies which invest in them, and also about the people that work in them.
                The database on which this website was built upon was sourced from <span style="font-weight: bold; color: #FFA500">Dealroom.co</span>,
                an intelligent global database to track innovative companies and identify growth opportunities.
                From there, we were able to collect datasets containing information on the Unicorns, and their respective investors. 
                The collected data included:
                <ul>
                    <li>The <span style="font-weight: bold; color: #FFA500">location</span> of the Unicorns and their Investors</li>
                    <li><span style="font-weight: bold; color: #FFA500">Foundation</span> <span style="font-weight: bold; color: #FFA500">year</span>, and the year in which they became a Unicorn</li>
                    <li>The <span style="font-weight: bold; color: #FFA500">Industries</span> in which the Unicorns operate</li>
                    <li><span style="font-weight: bold; color: #FFA500">Background</span> information on the Investors</li>
                    <li>And much more!</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        fig11 = home_page_map(active_unicorn_count, operational_unicorns, industry_colors)
        st.plotly_chart(fig11, use_container_width=True)
        
        
        

        
        


        
        
    elif page == "The Market Evolution":
        st.markdown("""
            <div style='text-align: center'>
                <h1>The Evolution of Unicorn Ecosystems</h1>
            </div>
            """, unsafe_allow_html=True)
        css = """
        <style>
            .blur-box {
                background-color: rgba(0, 0, 0, 0.5);  /* White background with 50% opacity */
                backdrop-filter: blur(8px);  /* Blur effect */
                border-radius: 10px;  /* Rounded corners */
                padding: 20px;  /* Padding inside the box */
                margin: 10px;  /* Margin around the box */
                color: white;  /* Text color */
                font-size: 16px;  /* Font size */
            }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
        st.header("The Market Evolution")
        st.markdown("""
        <style>
            /* Adjusts the padding and maximum width of the main container */
            .appview-container .main .block-container {
                max-width: 100%;  /* Full width */
                padding-top: 1rem;  /* Top padding */
                padding-right: 3rem;  /* Right padding */
                padding-left: 2rem;  /* Left padding */
                padding-bottom: 1rem;  /* Bottom padding */
            }
            /* Optionally hides footer and file uploader button if not needed */
            .uploadedFile { display: none; }
            footer { visibility: hidden; }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        with col1:
            fig12 = evolution_vcs_uni(unicorn_df, vcs)
            st.plotly_chart(fig12, use_container_width=True)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(""" <div class='blur-box'>
                        <br>
                        <br>
                        <br>
                        <br>
                        What determines the <span style="font-weight: bold; color: #FFA500">developments</span> of Unicorns? And why is this phenomenon so recent?
                        From our analysis, it emerges that one of the predominant factors for the development of an efficient entrepreneurial ecosystem is the <span style="font-weight: bold; color: #FFA500">availability</span>
                        <span style="font-weight: bold; color: #FFA500">of</span> <span style="font-weight: bold; color: #FFA500">funds</span>. 
                        These have not been easy to obtain for startups, as they represent too risky entities for traditional credit intermediaries. In fact, usually, in the initial stages, startups have a balance sheet situation that could be summarized as follows: high costs for research and development and very low, if not zero, revenues. 
                        It was therefore necessary to define sources of credit other than banking: <span style="font-weight: bold; color: #FFA500">Ventur</span> <span style="font-weight: bold; color: #FFA500">Capital</span>.
                        Observing the trend of players, it is clear how Venture Capitals have had a <span style="font-weight: bold; color: #FFA500">driving</span> <span style="font-weight: bold; color: #FFA500">effect</span> on unicorns. These have only surpassed the former in the last decade.
                        Certainly, this alone was not the reason why such an exponential increase in the number of unicorns has been recorded recently, but let's not rush too much!
                        <br>
                        <br>
                        <br>
                        <br>
                        </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.header("  A Global Outlook on the Rise of Venture Capitals and Unicorns")
        st.markdown(""" <div class='blur-box'>
                    The relationship between the availability of funds and the development of an efficient entrepreneurial
                    ecosystem becomes even more evident when observing the geographical distribution of unicorns and venture capital.
                    </div>""", unsafe_allow_html=True)
        fig3 = plot_vc_vs_unicorn_map(df_vc_extended, df_un_extended)
        st.plotly_chart(fig3, use_container_width=True)
            
        st.markdown(f"""<div class='blur-box'>
                    Until the early 1990s, we can see that the launching of startups is <span style="font-weight: bold; color: #FFA500">entirely</span> <span style="font-weight: bold; color: #FFA500">concentrated</span> in the two areas of United States 
                    and few European nations such as England, France, and Spain.
                    From the early 2000s onward, thanks also to the increase in the phenomenon of <span style="font-weight: bold; color: #FFA500">globalization</span>, there was an exponential growth in the number of foundations, with a significant spread and coverage of almost all areas of the world,
                    but always maintaining the beating heart in the North American, Europe and Chinese areas. Particularly, China was the state that has recorded the largest number of Unicorns foundations since 2000 (immediately after the United States),
                    testifying how the relentless drive toward the development of new technologies and globalization reflects the China development policy focused on innovation and openness to the international market.
                    <br>
                    <br>
                    Overall, this scenario is showing us how Unicorns' Startups tends to be launched in areas with a <span style="font-weight: bold; color: #FFA500">massive</span> <span style="font-weight: bold; color: #FFA500">presence</span> of Venture Capital companies. 
                    Indeed, in almost all areas of the world we can see that when there was an increasing in Venture Capital companies,
                    few years later there will be a substantial concentration of the founding of Unicorns in the same area, reflecting the <span style="font-weight: bold; color: #FFA500">close</span> <span style="font-weight: bold; color: #FFA500">correlation</span> between these two phenomena. 
                    The only exception we could noticed was in East Asia, where the launch of numerous Unicorns is not closely linked to the presence of Venture Capital companies, but would seem to be related to other factors.
                    </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.header("The Journey to Becoming a Unicorn")
        st.markdown(""" <div class='blur-box'>
                    The increasing relevance of the unicorn phenomenon is also due to the <span style="font-weight: bold; color: #FFA500">growing</span> <span style="font-weight: bold; color: #FFA500">dynamism</span> of this environment. 
                    In fact, if at the beginning of the phenomenon the average time to become a unicorn was 37 years, today it is much shorter. But why? 
                    The responsibility for this phenomenon can be attributed to a <span style="font-weight: bold; color: #FFA500">multitude</span> <span style="font-weight: bold; color: #FFA500">of</span>
                    <span style="font-weight: bold; color: #FFA500">forces</span> that have influenced the evolution of the market, and in this section,
                    we will try to analyze the main ones.
                    </div>""", unsafe_allow_html=True)
        
        fig1 = plot_unicorn_vc_trends(unicorn_and_vc_per_year, average_time, average_time_general, industry_colors)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(""" <div class='blur-box'>
                        As mentioned earlier, the increase in the <span style="font-weight: bold; color: #FFA500">availability</span> <span style="font-weight: bold; color: #FFA500">funds</span> has certainly played a key role.
                        The development of alternative sources of credit, not only venture capital but also crowdfunding, for example,
                        has allowed for a more rapid development of the entrepreneurial ecosystem.
                        <br>
                        <br>
                        The increasing <span style="font-weight: bold; color: #FFA500">globalization</span> and <span style="font-weight: bold; color: #FFA500">interconnectedness</span> of markets have allowed startups to tap into a larger user base. 
                        <br>
                        Simultaneously, by homogenizing consumer tastes, it has expanded the potential market for startups.
                        <br>
                        The development of an entrepreneurial ecosystem has not only led to the founding of new startups and venture capital firms but also support structures such as <span style="font-weight: bold; color: #FFA500">startup</span> <span style="font-weight: bold; color: #FFA500">incubators</span>.
                        Their role is precisely to accelerate the development of new companies by assisting them both in terms of idea validation and in building networks between founders and potential investors.
                        <br>
                        Furthermore, an additional accelerating effect has been brought about by changes in the <span style="font-weight: bold; color: #FFA500">business</span> <span style="font-weight: bold; color: #FFA500">model</span>. 
                        Previously, growth was also tied to an increase in fixed costs, but now, thanks to agile growth, startups can grow without increasing fixed costs by adapting to market fluctuations.
                        <br>
                        <br>
                        In conclusion, the rapid advancements in <span style="font-weight: bold; color: #FFA500">technology</span>, particularly in computing power, internet infrastructure, and software development tools, have significantly accelerated the pace of innovation.
                        This has enabled startups to develop and scale their products or services much more quickly than in the past.
                    </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.header("Race to the Top: The Rise of the most Proominent Industries")
        fig2 = plot_time_to_unicorn(merged_data, industry_colors)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(""" <div class='blur-box'>
                        Since the 1980s finance and real estate, technology and software, and health and wellness have steadily ascended to the pinnacle of unicorn creation,
                        driven by a confluence of factors and a fascinating evolutionary process.
                        <br>
                        <br>
                        In the realm of finance and real estate, the seeds of transformation were sown with the advent of <span style="font-weight: bold; color: #FFA500">computerizazion</span>.
                        As algorithms and digital databases began to replace ledger books and filing cabinets, the industry underwent a seismic shift towards efficiency and innovation. With the rise of the internet in the 1990s, 
                        online banking and e-commerce revolutionized how we interacted with money and property, laying the groundwork for future unicorn giants.
                        <br>
                        <br>
                        Meanwhile, the technology and software sector embarked on a journey of exponential growth and disruption.
                        The 1980s saw the birth of <span style="font-weight: bold; color: #FFA500">personal</span> <span style="font-weight: bold; color: #FFA500">computing</span>, with pioneers like Microsoft and Apple leading the charge.
                        As the <span style="font-weight: bold; color: #FFA500">internet</span> became mainstream in the 1990s, a new breed of startups emerged, leveraging connectivity to create revolutionary products and services.
                        From the dot-com boom to the rise of social media and mobile apps, the trajectory of this sector was propelled by a relentless pursuit of innovation and a willingness to embrace change.
                        <br>
                        <br>
                        In parallel, the health and wellness industry experienced a paradigm shift driven by <span style="font-weight: bold; color: #FFA500">scientific</span> <span style="font-weight: bold; color: #FFA500">advancements</span> and changing consumer attitudes.
                        Breakthroughs in biotechnology and pharmaceuticals paved the way for personalized medicine, while a growing awareness of the importance of self-care fueled demand for wellness products and services. 
                        From fitness trackers to mindfulness apps, the sector embraced technology as a means to empower individuals to take control of their health and well-being.
                        <br>
                        <br>
                        Throughout this journey, these sectors have remained at the forefront of unicorn creation, fueled by a potent <span style="font-weight: bold; color: #FFA500">combination</span> <span style="font-weight: bold; color: #FFA500">of</span> 
                        <span style="font-weight: bold; color: #FFA500">innovation</span>, <span style="font-weight: bold; color: #FFA500">investement</span>, <span style="font-weight: bold; color: #FFA500">and</span> <span style="font-weight: bold; color: #FFA500">market</span> 
                        <span style="font-weight: bold; color: #FFA500">demand</span>. With each passing decade, the pace of change accelerates, ushering in new opportunities and challenges for entrepreneurs bold enough to seize them.
                        As we look to the future, one thing remains clear: the reign of finance and real estate, technology and software, and health and wellness as the top unicorn-producing sectors shows no signs of waning.
                    """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        
        
        
    elif page == "Investments":
        st.markdown("""
            <div style='text-align: center'>
                <h1>The Evolution of Unicorn Ecosystems</h1>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <style>
            /* Adjusts the padding and maximum width of the main container */
            .appview-container .main .block-container {
                max-width: 100%;  /* Full width */
                padding-top: 1rem;  /* Top padding */
                padding-right: 3rem;  /* Right padding */
                padding-left: 2rem;  /* Left padding */
                padding-bottom: 1rem;  /* Bottom padding */
            }
            /* Optionally hides footer and file uploader button if not needed */
            .uploadedFile { display: none; }
            footer { visibility: hidden; }
        </style>
        """, unsafe_allow_html=True)
        css = """
        <style>
            .blur-box {
                background-color: rgba(0, 0, 0, 0.5);  /* White background with 50% opacity */
                backdrop-filter: blur(8px);  /* Blur effect */
                border-radius: 10px;  /* Rounded corners */
                padding: 20px;  /* Padding inside the box */
                margin: 10px;  /* Margin around the box */
                color: white;  /* Text color */
                font-size: 16px;  /* Font size */
            }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
        
        st.title('Total VC Funding and Exit Amounts Totalled by Region')
        # Handling VC funding visualization
        st.header('VC Funding by Region')
        continent_select_vc = st.selectbox("Select a continent and press 'Show VC Details', to check the distribution within the Continent:", continent_data_vc['Continent'].unique(), key='vc_select')
        if st.button('Show VC Details'):
            st.session_state.selected_continent_vc = continent_select_vc
        if st.button('Show All Continents for VC', key='vc_reset'):
            st.session_state.selected_continent_vc = None
        if 'selected_continent_vc' in st.session_state and st.session_state.selected_continent_vc:
            country_data = countries[countries['Continent'] == st.session_state.selected_continent_vc]
            country_data = country_data.sort_values(by='TOTAL VC FUNDING', ascending=False).reset_index(drop=True)
            fig5 = go.Figure(data=[
                go.Bar(
                    x=country_data['Country'],
                    y=country_data['TOTAL VC FUNDING'],
                    text=country_data['AMOUNT OF EXITS (2000)'],
                    textposition='outside',
                )
            ])
            title_text_vc=f'VC Funding in {st.session_state.selected_continent_vc}'
            fig5.update_layout(
                                       title = {
                                        'text': title_text_vc,  # Title text
                            'y':0.97,  # Position of the title (0.0 to 1.0, top to bottom)
                            'x':0.5,  # Position of the title (0.0 to 1.0, left to right, 0.5 is center)
                            'xanchor': 'center',  # Ensures the title is centered at the x position
                            'yanchor': 'top',  # Anchors the title to the top of the plot
                            'font': {
                                'family': "Arial",  # Font family
                                'size': 24,  # Font size
                                'color': "orange"  # Font color
                    }
            },
                               xaxis_title="Country", 
                               yaxis_title="Total VC Funding",
                                plot_bgcolor='black',  # Background color of the plotting area
                                paper_bgcolor='black',  # Background color of the surrounding paper
                                width=1200,  # Width of the plot
                                height=700  # Height of the plot
                                )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            fig5 = create_continent_VC_plot(continent_data_vc)
            st.plotly_chart(fig5, use_container_width=True)
        st.markdown(""" <div class='blur-box'>
                        This graph shows the how much VC Funding has occurred in each <span style="font-weight: bold; color: #FFA500">continent</span>, and every <span style="font-weight: bold; color: #FFA500">country</span> 
                        belonging to that continent, since the <span style="font-weight: bold; color: #FFA500">year</span> <span style="font-weight: bold; color: #FFA500">2000</span>! Like in most of the graphs, 
                        the North American giants dominates the rest of the world when it comes to funding, 
                        however, Canada makes up only a small slice of that funding. 
                        In Europe, the UK represents most of the funding, but when looking at the bigger picture, the distribution is much wider than in other continents; however, 
                        the Western and Northern Europe has clearly received much more funding than its Eastern Counterpart; however, the funding in Eastern Europe is sure to increase in coming years,
                        with countries such as Estonia being dubbed a “start-up haven”, with nearly 400 start-ups per million habitants!
                        </div>""", unsafe_allow_html=True)
        # Handling Exit Amounts visualization
        st.header('Exit Amounts by Region')
        continent_select_exit = st.selectbox("Select a continent and press 'Show Exit Details', to check the distribution within the Continent:", continent_data_exit['Continent'].unique(), key='exit_select')
        if st.button('Show Exit Details'):
            st.session_state.selected_continent_exit = continent_select_exit
        if st.button('Show All Continents for Exits', key='exit_reset'):
            st.session_state.selected_continent_exit = None
        if 'selected_continent_exit' in st.session_state and st.session_state.selected_continent_exit:
            country_data = countries[countries['Continent'] == st.session_state.selected_continent_exit]
            country_data = country_data.sort_values(by='TOTAL EXIT AMOUNTS', ascending=False).reset_index(drop=True)
            fig6 = go.Figure(data=[
                go.Bar(
                    x=country_data['Country'],
                    y=country_data['TOTAL EXIT AMOUNTS'],
                    text=country_data['AMOUNT OF EXITS (2000)'],
                    textposition='outside'
                )
            ])
            title_text_exit=f'Exit Amounts in {st.session_state.selected_continent_exit}'
            fig6.update_layout(
                            title = {
                                'text': title_text_exit,  # Title text
                                'y':0.97,  # Position of the title (0.0 to 1.0, top to bottom)
                                'x':0.5,  # Position of the title (0.0 to 1.0, left to right, 0.5 is center)
                                'xanchor': 'center',  # Ensures the title is centered at the x position
                                'yanchor': 'top',  # Anchors the title to the top of the plot
                                'font': {
                                'family': "Arial",  # Font family
                                'size': 24,  # Font size
                                'color': "orange"  # Font color
                                         }
                            }, 
                               xaxis_title="Country", 
                               yaxis_title="Total Exit Amounts",
                                plot_bgcolor='black',  # Background color of the plotting area
                    paper_bgcolor='black',  # Background color of the surrounding paper
                    width=1200,  # Width of the plot
                    height=700  # Height of the plot
            )
            st.plotly_chart(fig6, use_container_width=True)
        else:
            fig6 = create_continent_exit_plot(continent_data_exit)
            st.plotly_chart(fig6, use_container_width=True)
        st.markdown(""" <div class='blur-box'>
                    This graph highlights the cumulative monetary amount generated by exits in every <span style="font-weight: bold; color: #FFA500">continent</span> and <span style="font-weight: bold; color: #FFA500">country</span> since 
                    the <span style="font-weight: bold; color: #FFA500">year</span> <span style="font-weight: bold; color: #FFA500">2000</span>! For unicorns, an exit signifies when the company is either <span style="font-weight: bold; color: #FFA500">acquired</span> by a much larger company,
                    or when the company becomes publicly traded on stock markets via an <span style="font-weight: bold; color: #FFA500">IPO</span>. Once again, North America, and specifically the United States dominates this sphere. The European graphic also follows a nearly identical distribution to
                    the aforementioned distribution in the VC Funding graphic.
                    <br>
                    <br>
                    It's interesting to compare the two graphs on VC funding and exits between the countries and continents. 
                    While the two graphs roughly follow the same distribution, there are some exceptions to the rule. For instance, 
                    while Asia has received more funding than Europe, Europe far exceeds them in Exits! Moreover, when confronting the VC Funding and Exits generated for Europe, 
                    certain countries like Moldova generated only $1.1M in funding, and yet still accumulated $17M in exits which is a better return on investment than the giants of Europe, the UK!
                    </div>""", unsafe_allow_html=True)
        st.header('Exit Amounts by Industry')
        fig8 = for_industries(industries)
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown(""" <div class='blur-box'>
                    <span style="font-weight: bold; color: #FFA500">Technology</span> <span style="font-weight: bold; color: #FFA500">Industry</span> nearly had a cumulative amount of exits as large as the next four industries! 
                    This also correlates with insights from the founders’ backgrounds, as Technology studies were the most popular amongst founders.
                    In any case, graphs like this clearly demonstrate that we’re in a <span style="font-weight: bold; color: #FFA500">digital</span> <span style="font-weight: bold; color: #FFA500">age</span>. When looking at the 
                    world around us, everything we do is completely dependent on technology nowadays, and consequently, that clearly had an effect on the start-ups!
                    </div>""", unsafe_allow_html=True)
        
        st.header('Exit Amounts by Investors')
        fig9 = for_investors(total_exits)
        st.plotly_chart(fig9, use_container_width=True)
        st.markdown(""" <div class='blur-box'>
                    The investors which profited the most from their investments were mainly <span style="font-weight: bold; color: #FFA500">large</span> <span style="font-weight: bold; color: #FFA500">financial</span> 
                    <span style="font-weight: bold; color: #FFA500">institutions</span> from the United States.
                    The largest one was Sequoia capital, a venture capital firm located in Paolo Alto, California, right in the heart of Silicon Valley, who amassed 12T through their investments in 
                    AirBnB, Google, and WhatsApp amongst many others. The only <span style="font-weight: bold; color: #FFA500">one</span> <span style="font-weight: bold; color: #FFA500">investor</span> who snuck into the top 10 was Softbank, a Japanese multinational.
                    </div>""", unsafe_allow_html=True)
        
        st.header('Most Profitable Investors')
        
        col1, col2 = st.columns([3, 1])
        with col1:
            fig10 = cluster_bubble_chart(total_exits)
            st.plotly_chart(fig10, use_container_width=True)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(""" <div class='blur-box'>
                        <br>
                        <br>
                        <br>
                        <br>
                        <br>
                        <br>
                        This graph shows the <span style="font-weight: bold; color: #FFA500">clustered</span> exit amounts by investors. The size of the bubbles represents the exit amount, 
                        and the color represents the amount of exits. The graph is clustered by the location of the investors, 
                        and it is clear that the United States has the most investors with the highest exit amounts.
                        <br>
                        <br>
                        <br>
                        <br>
                        <br>
                        <br>
                        <br>
                        <br>
                        </div>""", unsafe_allow_html=True)
    
    elif page == "Investments Flow":
        st.markdown("""
            <div style='text-align: center'>
                <h1>The Evolution of Unicorn Ecosystems</h1>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <style>
            /* Adjusts the padding and maximum width of the main container */
            .appview-container .main .block-container {
                max-width: 100%;  /* Full width */
                padding-top: 1rem;  /* Top padding */
                padding-right: 3rem;  /* Right padding */
                padding-left: 2rem;  /* Left padding */
                padding-bottom: 1rem;  /* Bottom padding */
            }
            /* Optionally hides footer and file uploader button if not needed */
            .uploadedFile { display: none; }
            footer { visibility: hidden; }
        </style>
        """, unsafe_allow_html=True)
        css = """
        <style>
            .blur-box {
                background-color: rgba(0, 0, 0, 0.5);  /* White background with 50% opacity */
                backdrop-filter: blur(8px);  /* Blur effect */
                border-radius: 10px;  /* Rounded corners */
                padding: 20px;  /* Padding inside the box */
                margin: 10px;  /* Margin around the box */
                color: white;  /* Text color */
                font-size: 16px;  /* Font size */
            }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
        st.title("Unicorn Investment Flow Dashboard")
        st.markdown(""" <div class='blur-box'>
                    Here you’ll be able to see how the <span style="font-weight: bold; color: #FFA500"> flow of investments </span> throughout time and region between the Investors and the Unicorns! 
                    Please feel free to interact with the graph by moving the slider around, or filtering by region! Furthermore, if any Investor or Unicorn piques your interest,
                    you’ll be able to see where they send money to, or where they receive money by hovering your pointer over your point of interest!
                    <br>
                    <br>
                    The effects of  <span style="font-weight: bold; color: #FFA500"> globalization </span> are made blatantly obvious by this graphic. While the flow of investments between investor and unicorn have short initial distances, 
                    either limited by region, or at most, between North America and Europe, as time progresses, one could see how the distance between investors and their respective unicorns increases.
                    For example, in 2011, Bain Capital, an American giant of the private investment world headquartered in Boston, invested in a Unicorn called MYOB in Melbourne Australia, 
                    a distance of nearly 17,000km between the two companies’ respective headquarters.
                    <br>
                    <br>
                    However, even though the distances between investments only seem to be increasing, the distances themselves have never seemed smaller. 
                    Due to the advancements in technology, and very likely, due to advancements made by the companies being invested in on this graph, like X (i.e., Twitter) for example, 
                    communication between the two parties has never been easier, and is likely the reason that these investments can occur. This can cause a  <span style="font-weight: bold; color: #FFA500">butterfly effect</span>,
                    whereby people become inspired by others to start their own company after seeing that gathering funds is possible, which in turn creates more Unicorns, and so-on and so forth. While just a hypothesis,
                    it could very well explain why the number of unicorns and number of regions increase as time progresses, for the dense and bare areas.
                    </div>""", unsafe_allow_html=True)

        # Dropdown to select region
        selected_region = st.selectbox(
            "Select a region",
            options=unicorninvestors['UNICORN REGION'].unique(),
            index=0
        )

        # Slider to select year
        selected_year = st.slider(
            "Select a year",
            min_value=int(unicorninvestors['YEAR OF INVESTMENT'].min()),
            max_value=int(unicorninvestors['YEAR OF INVESTMENT'].max()),
            value=int(unicorninvestors['YEAR OF INVESTMENT'].min()),
            format="%d"
        )

        # Function to update the figure based on selections
        def update_figure(selected_region, selected_year):
            filtered_data = unicorninvestors[
                (unicorninvestors['UNICORN REGION'] == selected_region) &
                (unicorninvestors['YEAR OF INVESTMENT'] == selected_year)
            ]

            fig = go.Figure()

            if not filtered_data.empty:
                # Plotting lines connecting investors and unicorns, exclude from legend
                lon_lines = [
                    lon for pair in zip(filtered_data['INVESTOR LONGITUDE'], filtered_data['UNICORN LONGITUDE'], [None]*len(filtered_data)) for lon in pair
                ]
                lat_lines = [
                    lat for pair in zip(filtered_data['INVESTOR LATITUDE'], filtered_data['UNICORN LATITUDE'], [None]*len(filtered_data)) for lat in pair
                ]
                fig.add_trace(go.Scattergeo(
                    mode="lines",
                    lon=lon_lines,
                    lat=lat_lines,
                    line=dict(width=2, color='black'),
                    hoverinfo='none',
                    showlegend=False  # Exclude this trace from the legend
                ))

                # Adding investor markers
                investor_hover_text = [
                    f"{name} invests in: {', '.join(set(filtered_data[filtered_data['INVESTOR LONGITUDE'] == lon]['UNICORN']))}"
                    for name, lon in zip(filtered_data['INVESTORS'], filtered_data['INVESTOR LONGITUDE'])
                ]
                fig.add_trace(go.Scattergeo(
                    mode="markers",
                    lon=filtered_data['INVESTOR LONGITUDE'],
                    lat=filtered_data['INVESTOR LATITUDE'],
                    marker=dict(size=10, color='blue', line=dict(width=1)),
                    name='Investors',
                    text=investor_hover_text,
                    hoverinfo='text'
                ))

                # Adding unicorn markers
                unicorn_hover_text = [
                    f"{name} receives investment from: {', '.join(set(filtered_data[filtered_data['UNICORN LONGITUDE'] == lon]['INVESTORS']))}"
                    for name, lon in zip(filtered_data['UNICORN'], filtered_data['UNICORN LONGITUDE'])
                ]
                fig.add_trace(go.Scattergeo(
                    mode="markers",
                    lon=filtered_data['UNICORN LONGITUDE'],
                    lat=filtered_data['UNICORN LATITUDE'],
                    marker=dict(size=10, color='red', line=dict(width=1)),
                    name='Unicorns',
                    text=unicorn_hover_text,
                    hoverinfo='text'
                ))
            else:
                # Add a basic plot to show the world map
                fig.add_trace(go.Scattergeo(showlegend=False))
                fig.update_geos(
                    projection_type="natural earth",
                    showland=True,
                    landcolor="lightgrey"
                )
            title=f"Global Investment Flows for Unicorns in {selected_year}"
            fig.update_layout(
                title = {
                    'text': title,  # Title text
                    'y':0.97,  # Position of the title (0.0 to 1.0, top to bottom)
                    'x':0.5,  # Position of the title (0.0 to 1.0, left to right, 0.5 is center)
                    'xanchor': 'center',  # Ensures the title is centered at the x position
                    'yanchor': 'top',  # Anchors the title to the top of the plot
                    'font': {
                        'family': "Arial",  # Font family
                        'size': 24,  # Font size
                        'color': "orange"  # Font color
                             }
            },
                geo = dict(
                    scope='world',
                    showland=True,
                    landcolor='rgb(0, 100, 0)',
                    showcountries=True,
                    countrycolor='rgb(0, 0, 0)',
                    showocean=True,
                    oceancolor='rgb(204, 204, 255)',
                    showcoastlines=True,
                    coastlinecolor='rgb(0,0,0)',
                    showframe=True,
                    framecolor='rgb(0,0,0)',
                    bgcolor='rgb(0, 0, 0)',
                    projection=dict(
                    type='natural earth')
                ),
                plot_bgcolor='black',  # Background color of the plotting area
                paper_bgcolor='black',  # Background color of the surrounding paper
                width=1250,  # Width of the plot
                height=750  # Height of the plot

            )

            return fig

        # Display the figure
        fig = update_figure(selected_region, selected_year)
        st.plotly_chart(fig, use_container_width=True)
            
    
    

    elif page == "Backgrounds":
        st.markdown("""
            <div style='text-align: center'>
                <h1>The Evolution of Unicorn Ecosystems</h1>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <style>
            /* Adjusts the padding and maximum width of the main container */
            .appview-container .main .block-container {
                max-width: 100%;  /* Full width */
                padding-top: 1rem;  /* Top padding */
                padding-right: 3rem;  /* Right padding */
                padding-left: 2rem;  /* Left padding */
                padding-bottom: 1rem;  /* Bottom padding */
            }
            /* Optionally hides footer and file uploader button if not needed */
            .uploadedFile { display: none; }
            footer { visibility: hidden; }
        </style>
        """, unsafe_allow_html=True)
        css = """
        <style>
            .blur-box {
                background-color: rgba(0, 0, 0, 0.5);  /* White background with 50% opacity */
                backdrop-filter: blur(8px);  /* Blur effect */
                border-radius: 10px;  /* Rounded corners */
                padding: 20px;  /* Padding inside the box */
                margin: 10px;  /* Margin around the box */
                color: white;  /* Text color */
                font-size: 16px;  /* Font size */
            }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
        
        st.header("Universities to Attend if You Want to be a Founder")
        fig7 = best_unis_investors(founders_edu)
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown(""" <div class='blur-box'> 
                    When looking at which universities founders most attended, there is a clear dominance from the <span style="font-weight: bold; color: #FFA500">United</span> 
                    <span style="font-weight: bold; color: #FFA500">States</span>; 
                    however, there are some additions such as the Indian Institute of Technology, 
                    and the Tel Aviv university, showing that Unicorns are a worldwide phenomenon.
                    </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.header("Founders Academic Backgrounds")
        fig8 = what_funders_studied(founders_edu)
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown(""" <div class='blur-box'> 
                        When looking at what the founders studied, <span style="font-weight: bold; color: #FFA500">technical</span> <span style="font-weight: bold; color: #FFA500">backgrounds</span> had the largest, 
                        followed by <span style="font-weight: bold; color: #FFA500">business</span> and <span style="font-weight: bold; color: #FFA500">IT</span>. This fact highlights recent trends that most start-ups are tech based,
                        but that simply having tech knowledge is not enough to create a successful company. Investors need business knowledge as well, 
                        especially when managing their companies, and when interacting with investors.
                        </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.header("Are They First-Time Founders ?")
        col1, col2 = st.columns([3, 1])
        with col1:
            fig9 = first_company(founders)
            st.plotly_chart(fig9, use_container_width=True)
        with col2:
            st.markdown(""" <div class='blur-box'> 
                        <br>
                        <br>
                        <br>
                        <br>
                        <br>
                        Surprisingly, these unicorns were the first entrepreneurial exploit for <span style="font-weight: bold; color: #FFA500">over</span> <span style="font-weight: bold; color: #FFA500">80%</span> of founders. 
                        This just goes to show that while failure may be the best teacher, it isn’t always needed!
                        <br>
                        <br>
                        <br>
                        <br>
                        <br>
                        <br>
                        </div>""", unsafe_allow_html=True)
        
        
        
        
    elif page == "The Evolution of Womens' Roles":
        st.markdown("""
            <div style='text-align: center'>
                <h1>The Evolution of Unicorn Ecosystems</h1>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <style>
            /* Adjusts the padding and maximum width of the main container */
            .appview-container .main .block-container {
                max-width: 100%;  /* Full width */
                padding-top: 1rem;  /* Top padding */
                padding-right: 3rem;  /* Right padding */
                padding-left: 2rem;  /* Left padding */
                padding-bottom: 1rem;  /* Bottom padding */
            }
            /* Optionally hides footer and file uploader button if not needed */
            .uploadedFile { display: none; }
            footer { visibility: hidden; }
        </style>
        """, unsafe_allow_html=True)
        css = """
        <style>
            .blur-box {
                background-color: rgba(0, 0, 0, 0.5);  /* White background with 50% opacity */
                backdrop-filter: blur(8px);  /* Blur effect */
                border-radius: 10px;  /* Rounded corners */
                padding: 20px;  /* Padding inside the box */
                margin: 10px;  /* Margin around the box */
                color: white;  /* Text color */
                font-size: 16px;  /* Font size */
            }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
        st.markdown("""
                    <style>
                        .css-1lcbmhc {
                            filter: blur(8px);
                        }
                    </style>
                    """, unsafe_allow_html=True)
        
        st.header("The Unfolding of Womens\' Presence in the Unicorn Ecosystem")


        st.markdown(""" <div class='blur-box'>
                    The role of women in venture capital (VC) and the unicorn ecosystem has experienced a <span style="font-weight: bold; color: #FFA500">notable</span> <span style="font-weight: bold; color: #FFA500">uptick</span> since the 
                    1980s, driven by a <span style="font-weight: bold; color: #FFA500">combination of societal shifts, advocacy efforts, and recognition of untapped potential</span>. 
                    Historically, VC and unicorn creation were dominated by male investors and founders, but as awareness of gender diversity's benefits grew, so did efforts to level the playing field. <span style="font-weight: bold; color: #FFA500">Initiatives promoting women</span> in STEM fields, 
                    mentorship programs, and the rise of female-led investment firms have all played pivotal roles in expanding opportunities for women. Furthermore, research showcasing the strong performance of companies with diverse leadership teams has underscored the business case for inclusion.
                    Today, while disparities persist, 
                    strides have been made, with more women securing leadership roles in VC firms and founding unicorn startups, contributing to a more equitable and innovative entrepreneurial landscape.
                    </div>""", unsafe_allow_html=True)
        fig_3 = plot_gender_analysis(vc_genders, unicorn_gender, available_years)
        st.plotly_chart(fig_3, use_container_width=True)
        
        
    elif page == "About Us":
        st.markdown("""
            <div style='text-align: center'>
                <h1>About Us</h1>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <style>
            /* Adjusts the padding and maximum width of the main container */
            .appview-container .main .block-container {
                max-width: 100%;  /* Full width */
                padding-top: 1rem;  /* Top padding */
                padding-right: 3rem;  /* Right padding */
                padding-left: 2rem;  /* Left padding */
                padding-bottom: 1rem;  /* Bottom padding */
            }
            /* Optionally hides footer and file uploader button if not needed */
            .uploadedFile { display: none; }
            footer { visibility: hidden; }
        </style>
        """, unsafe_allow_html=True)
        
        import base64
        def image_to_data_uri(file_path):
            """
            Converts an image file to a data URI.
            """
            with open(file_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:image/png;base64,{encoded_string}"
        
        class PersonInfoMarkdown:
            def __init__(self, image_data_uri, full_name, student_id, linkedin_url):
                self.image_data_uri = image_data_uri
                self.full_name = full_name
                self.student_id = student_id
                self.linkedin_url = linkedin_url

            def generate_html(self):
                html_content = f"""
                <div style="background-color: rgba(0, 0, 0, 0.5); 
                    backdrop-filter: blur(8px); 
                    border-radius: 10px;
                    padding:20px; 
                    text-align: center;">
                    <div style="margin: 10px; width: 150px; height: 150px; background-image: url('{self.image_data_uri}');
                    background-size: cover; border-radius: 50%; margin-left: auto; margin-right: auto;">
                    </div>
                    <h1>{self.full_name}</h1>
                    <h3>{self.student_id}</h3>
                    <a href="{self.linkedin_url}" target="_blank">LinkedIn Profile</a>
                </div>
                """
                return html_content

            def display(self, st):
                st.markdown(self.generate_html(), unsafe_allow_html=True)


            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            image_raf = image_to_data_uri("/Users/fab/Desktop/UNICORN_ANALYSIS/data_viz_presentation/Raf.jpeg") ### PATHS TO REPLACE
            info_raf = PersonInfoMarkdown (image_raf,
                                       'Raffaele Torelli',
                                       '775831',
                                       'https://www.linkedin.com/in/raffaele-torelli-04ab72242/')
            info_raf.display(st)
        
        with col2:
            image_dan = image_to_data_uri("/Users/fab/Desktop/UNICORN_ANALYSIS/data_viz_presentation/Dani.jpeg") ### PATHS TO REPLACE
            info_dan = PersonInfoMarkdown (image_dan,
                                       'Daniele De Robertis',
                                       '787291',
                                       'https://www.linkedin.com/in/daniele-de-robertis-4a9993297/')
            info_dan.display(st)
        with col3:
            image_dave = image_to_data_uri("/Users/fab/Desktop/UNICORN_ANALYSIS/data_viz_presentation/david.jpeg") ### PATHS TO REPLACE
            info_dave = PersonInfoMarkdown (image_dave,
                                       'David Paquette',
                                       '789331',
                                       'https://www.linkedin.com/in/dpaquette1999/')
            info_dave.display(st)
        with col4:
            image_fab = image_to_data_uri("/Users/fab/Desktop/UNICORN_ANALYSIS/data_viz_presentation/fsb.jpeg") ### PATHS TO REPLACE
            info_fab = PersonInfoMarkdown (image_fab,
                                       'Fabrizio Borrelli',
                                       '789121',
                                       'https://www.linkedin.com/in/fabrizioborrelli/')
            info_fab.display(st)
            
        
if __name__ == "__main__":
    main()