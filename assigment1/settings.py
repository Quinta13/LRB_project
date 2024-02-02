from os import path

import torch

# PATHS
ROOT_DIR = str(path.abspath(path.join(__file__, "../")))

DATA_URL = "https://d3da1k6uo8tbjf.cloudfront.net/d613ecb6-9349-11ee-9228-d6d3907c63d6?response-content-disposition=inline%3B%20filename%3D%22sound_symbolism_data.csv%22%3B%20filename%2A%3DUTF-8%27%27sound_symbolism_data.csv&response-content-type=text%2Fcsv&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAS5PME4CTY5HJXGX6%2F20231208%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Date=20231208T150119Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=d66be97f9f65c7c056e5696e845d9c7a9f66947fad7b9f2a566d2d96582cce27&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kM2RhMWs2dW84dGJqZi5jbG91ZGZyb250Lm5ldC9kNjEzZWNiNi05MzQ5LTExZWUtOTIyOC1kNmQzOTA3YzYzZDY~cmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj1pbmxpbmUlM0IlMjBmaWxlbmFtZSUzRCUyMnNvdW5kX3N5bWJvbGlzbV9kYXRhLmNzdiUyMiUzQiUyMGZpbGVuYW1lJTJBJTNEVVRGLTglMjclMjdzb3VuZF9zeW1ib2xpc21fZGF0YS5jc3ZcdTAwMjZyZXNwb25zZS1jb250ZW50LXR5cGU9dGV4dCUyRmNzdlx1MDAyNlgtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2XHUwMDI2WC1BbXotQ3JlZGVudGlhbD1BS0lBUzVQTUU0Q1RZNUhKWEdYNiUyRjIwMjMxMjA4JTJGdXMtZWFzdC0yJTJGczMlMkZhd3M0X3JlcXVlc3RcdTAwMjZYLUFtei1EYXRlPTIwMjMxMjA4VDE1MDExOVpcdTAwMjZYLUFtei1FeHBpcmVzPTg2NDAwXHUwMDI2WC1BbXotU2lnbmVkSGVhZGVycz1ob3N0XHUwMDI2WC1BbXotU2lnbmF0dXJlPWQ2NmJlOTdmOWY2NWM3YzA1NmU1Njk2ZTg0NWQ5YzdhOWY2Njk0N2ZhZDdiOWYyYTU2NmQyZDk2NTgyY2NlMjciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3MDIxMzQwNzl9fX1dfQ__&Signature=kYxky1cVEql5ZYO-euEBqTQEWaC5kI4-XCSGtr8Wms-W687ddRNXgJyQMWgTaEvLPbPX~aMm3T7dOUygrMPdC-nWs4imUeZkhI0kLHzfOoy5tmTmyxJgGCzzbp4eeq02ti5QvxOqqzqd1VCCPC-grtxQ1TkMKVL0I-LVLq1g~gtv5rP3a~Na-lriuW64rbpKG7VGRC-KacrrS0QUtC-lOSkVXpiFsU9l30KRpCons9-rl~cARf2IzlLr-LBHdJEFD8ZqcoMQkpNlyQbpoOPUx7NNUzTgOtVpF471EhJaVI325pOdfwgB~BtYocMtMed~kkwiwjEOxi0~Y1NaTr47Hw__&Key-Pair-Id=K2BMZZDBFKKL41"
DATA_DIR_NAME = "data"
FILE_NAME = "sound_symbolism.csv"
DATA_DIR = path.join(ROOT_DIR, "data")

SMALL_WORDS_FILE      = path.join(DATA_DIR, "small.txt"     )
BIG_WORDS_FILE        = path.join(DATA_DIR, "big.txt"       )
SMALL_WORDS_FILE_TEST = path.join(DATA_DIR, "small_test.txt")
BIG_WORDS_FILE_TEST   = path.join(DATA_DIR, "big_test.txt"  )

# COLORS

VOWELS = ["a", "e", "i", "o", "u"]
VOWELS_COLORS = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a"]

HISTO_COLORS  = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5"]

SMALL_BIG_COLORS     = ["#8E44AD", "#2ECC71"]
NETWORK_COLORS       = ["#BB8FCE", "#ABEBC6"]
CORRECT_WRONG_COLORS = ["#3498DB", "#F39C12"]

# TRAINING
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

