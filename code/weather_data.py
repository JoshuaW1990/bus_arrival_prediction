# This file is used to collect the weather data
# copied from: https://gist.github.com/philshem/8864437#file-wunderground_historical-py


import requests
import csv

def get_precip(gooddate):
    urlstart = 'http://api.wunderground.com/api/d083880ff5428216/history_'
    urlend = '/q/NY/New_York.json'

    url = urlstart + str(gooddate) + urlend
    data = requests.get(url).json()
    for summary in data['history']['dailysummary']:
        result = [gooddate,summary['date']['year'],summary['date']['mon'],summary['date']['mday'], summary['fog'], summary['rain'], summary['snow']]
    return result

if __name__ == "__main__":
    from datetime import date
    from dateutil.rrule import rrule, DAILY

    a = date(2017, 1, 1)
    b = date(2017, 1, 31)

    result = [['date', 'year', 'month', 'day', 'fog', 'rain', 'snow']]
    for dt in rrule(DAILY, dtstart=a, until=b):
        result.append(get_precip(dt.strftime("%Y%m%d")))

    # export to the csv file
    with open('weather.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',')
        for item in result:
            spamwriter.writerow(item)
