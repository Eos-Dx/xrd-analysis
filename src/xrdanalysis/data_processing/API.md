Eos Dx API Documentation

Friday 7 July 2023

Anastasia Zaytseva | [azaytzeva@eosdx.com](mailto:jfriedman@eosdx.com)

# Overview
This Eos Dx API has been specifically designed to cater to the needs of analytics and lab teams, providing them with access to measurement files and the corresponding data that are essential for their analysis processes. By leveraging this API, teams can optimize their workflow and efficiently retrieve the crucial information required for their analytical endeavors.

## Key Features
1. Seamless Data Retrieval: The API offers a seamless and intuitive interface for retrieving measurement files and their corresponding data. By providing the necessary parameters and filters, teams can effortlessly access the specific files and data points they require for their analysis.

2. Customizable Filters: The API supports a wide range of customizable filters, allowing users to narrow down their search and retrieve the most relevant measurement files and data. Whether it's based on study, data range, or any other specific criteria, the API empowers users to tailor their queries to meet their exact requirements.

3. Efficient Query Execution: The API employs efficient query execution techniques to ensure optimal performance. By leveraging advanced data processing capabilities, the API minimizes response times and maximizes throughput, enabling users to retrieve their desired files and data in a timely manner.

# Getting started
This section provides instructions on how to make an API call using any Python environment.

## Before start
For using API, you will need to:

- Get permission to access the database.
- Get access key from the development team by contacting either Michael Solomin (<msolomin@eosdx.com>) or Anastasia Zaytseva (<azaytzeva@eosdx.com>).
- Make sure you are using Python version 3.0+

Note that the access key is active for **3 months** and then requires renewal.

## Example API-request

<table><tr><th colspan="1" rowspan="4" valign="top"><p>import requests </p><p> </p><p>form = { </p><p>'key': 'your-access-key', </p><p>'cancer_tissue': True, </p><p>'measurement_id': '< 3', </p><p>'measurement_date': '2023-04-07',</p><p></p><p>} </p><p> </p><p>resp = requests.post('http://api.eosdx.com/api/getmultiple',form) </p><p> </p><p>with open('path/to/folder/response.zip','+wb') as f: </p><p>f.write(resp.content) </p></th><th colspan="1" valign="top">Requests library import </th></tr>
<tr><td colspan="1" valign="top"><p>Form for POST-request with access key and parameters of filters </p><p></p><p></p></td></tr>
<tr><td colspan="1" valign="top">Transferring the form </td></tr>
<tr><td colspan="1" valign="top">Getting response and saving it to the Zip-archive</td></tr>
</table>
##
## API-request format
The form data payload is defined as a dictionary named 'form'. It contains several parameters:

- 'key': This parameter should be replaced with your actual access key.
- Other parameters: To filter parameter values in a request, you need to specify the filter name in quotation marks and use '<', '>', or '=' to filter the values.

Please be aware that the database only supports the 'YYY-MM-DD' format for dates.

The requests.post() method is used to send the POST request to the API endpoint http://api.eosdx.com/api/getmultiple. The 'form' dictionary is passed as the data parameter to include the form data in the request. After receiving the response, the code creates file named 'response.zip' in the specified by user directory.

Make sure to replace 'your-access-key' with your actual access key and provide the correct path to the folder where you want to save the response file. Available filters for a request can be accessed in the [Description file columns](#_description_file_columns) section of the document.
#
# Response structure
This section outlines the format and contents of the API response. In this section, we provide a detailed description of the components included in the API response, which enables users to access and analyze the measurement data and retrieve important information about each measurement.
##
## Response files
When making a request to the API, the response will be a Zip-archive containing the following components:

1. Measurement Files Folder

This folder contains the measurement files in .npy format. Each file is named after the corresponding measurement's ID. These files store the actual measurement data.

- `   `Format of files: .npy
- `   `File names: Measurement ID (measurement\_id)

2. Calibration Files Folder

This folder contains the calibration files in .npy format. Each file is named after the calibration measurement's ID. These files store the calibration data associated with a measurement.

- `   `Format of files: .npy
- `   `File names: Calibration Measurement ID (calibration\_measurement\_id)

3. Description File

This file contains detailed information about every measurement. It provides additional metadata and attributes associated with each measurement. The CSV file serves as a reference for understanding the measurements and their corresponding data.

- `   `Format: .csv
- `   `File Name: description.csv
## <a name="bookmark1"></a><a name="_description_file_columns"></a>Description file columns and available filters
The description file contains following columns about measurements, which can be useful for analysis and research:

|**Name** |**Description**|**Entity**|
| -: | :- | :- |
|**calibration\_measurement\_id**|Identifier for each calibration measurement.|Calibration measurement |
|**machine\_id**|Identifier of a machine used to perform the measurement.|Machine|
|**study\_id**|Identifier of a study associated with the measurement.|Study|
|**patient\_id**|Identifier of a patient for whom the measurement was taken.|Patient|
|**measurement\_id**|Unique identifier for each measurement.|Measurement|
|**s3\_link**|Link to a file stored in an S3 bucket.|Measurement|
|**cancer\_tissue** |Information if a tissue was cancerous or not. (1 – cancer, 0 – non-cancerous, None - unknown)|Measurement|
|**tissue\_type**|Type of tissue measured.  |Measurement|
|**tissue\_size**|Size of a tissue being measured.|Measurement|
|**exposure\_time**|Duration of the exposure during a measurement.|Measurement|
|**measurement\_date**|Date and time when a measurement was taken.|Measurement|
|**calibration\_manual\_distance**|Manually recorded distance during a measurement.|Calibration measurement|
|**code\_name**|Code name associated with a patient.|Patient|
|**cancer\_diagnosis**|Cancer diagnosis of a patient.|Patient|
|**study\_name**|This column stores the name of the study associated with the measurement.|Study|
|**target**|This column represents the target of the measurement.|Machine|
|**wavelength**|This column stores the wavelength used during the measurement.|Machine|
|**name**|Name of a machine which was used to|Machine|
|**calibration\_date**|Date and time when the calibration was performed.|Calibration measurement|
|**calculated\_distance**|Calculated distance during a measurement.|Calibration measurement|
|**orig\_file\_name**|Original file name associated with a measurement.|<p>Measurement</p><p></p>|
|**Is\_blind**|Blind data marker.|Patient|
