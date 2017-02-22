# Thie file is the script for downloading the history file for the bus time
import urllib




# define a function for downloading
def download_history_file(year, month, date_list):
    year = str(year)
    if month < 10:
        month = '0' + str(month)
    else:
        month = str(month)
    base_url = 'http://data.mytransit.nyc/bus_time/'
    url = base_url + year + '/' + year + '-' + month + '/'
    download_file = urllib.URLopener()
    for date in date_list:
        if date < 10:
            date = '0' + str(date)
        else:
            date = str(date)
        filename = 'bus_time_' + year + month + date + '.csv.xz'
        file_url = url + filename
        download_file.retrieve(file_url, filename)

if __name__ == "__main__":
    # configuration for the dates
    year = 2016
    month = 2
    date_list = range(1, 30)

    # download the file
    download_history_file(year, month, date_list)
    
