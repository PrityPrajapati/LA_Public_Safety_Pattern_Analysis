# Raw data

Put the LAPD file here as **Crime_Data_from_2020_to_2024.csv** (about 255 MB, 1,004,894 rows).
It is not included in the zip because of its size.

Source: City of Los Angeles Open Data, "Crime Data from 2020 to Present"
- Dataset page: https://data.lacity.org/d/2nrs-mtv8
- Direct CSV:  https://data.lacity.org/api/views/2nrs-mtv8/rows.csv?accessType=DOWNLOAD
  (the portal file may be newer and named differently; rename it to the name above)

You only need the raw file to re-run the pipeline. The Streamlit app works without it,
because everything it needs is already in app/app_data/ and models/.
