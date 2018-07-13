#!/usr/bin/env python3

from sklearn import datasets

print("Loading house prices..")
house_prices = datasets.load_boston()
print(house_prices.data)

print("Loading house targets...")
print(house_prices.target)

print("Loading fifth image...")
digits = datasets.load_digits()
print(digits.images[4])
