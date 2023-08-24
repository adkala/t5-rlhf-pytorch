import csv
import random

def getRowsFromCSV(filename='pricerunner_aggregate.csv'):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        rows = []
        for row in csv_reader:
            rows.append(row)
            line_count += 1
        print('Parsed %i rows from %s' % (line_count, filename))
        return rows

def cleanRows(rows):
    newRows = []
    for row in rows:
        newRows.append((row[1], row[4], row[6])) # listing, product, category
    return newRows

def getCategoryProductListingTree(cleaned_rows):
    d = {}
    for listing, product, category in cleaned_rows:
        if category not in d:
            d[category] = {}
        if product not in d[category]:
            d[category][product] = set()
        d[category][product].add(listing)
    return d

def getCategoryBrandProductTree(cleaned_rows):
    d = {}
    for _, product, category in cleaned_rows:
        if category not in d:
            d[category] = {}
        s = product.split()
        brand = s[0] if len(s) < 2 or '%s %s' % (s[0], s[1]) != 'General Electric' else 'General Electric'
        if brand not in d[category]:
            d[category][brand] = set()
        d[category][brand].add(product)
    return d

