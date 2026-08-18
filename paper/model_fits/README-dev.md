To copy select figure files:
```for DIRECTORY in $(cat figures.txt); do cp -r ${DIRECTORY} figures_copy; done```

To create `runs.tar.gz` files:
```tar -hczvf runs.tar.gz -T runs.txt```

To upload `runs.tar.gz` files:
```for DIRECTORY in *; do curl --upload-file ${DIRECTORY}/runs.tar.gz -H "Authorization: Bearer $ACCESS_TOKEN" https://zenodo.org/api/files/a056ad0d-033f-4d1c-85b6-6fad275e0a05/${DIRECTORY}.tar.gz | cat; done```

