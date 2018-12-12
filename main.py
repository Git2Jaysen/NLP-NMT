# coding: utf-8

import json
import data_utils

def main():
  data_utils.preprocessing(json.load(open("data/config.json")))

if __name__ == "__main__":
    main()
