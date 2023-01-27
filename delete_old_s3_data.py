from src.module.db.s3 import S3

from datetime import datetime, timedelta


BUKET_NAME = 'autocryptotrading'


if __name__ == '__main__':
    s3 = S3(BUKET_NAME)
    two_month_ago = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d%H%M')
    for key in s3.s3.Bucket(BUKET_NAME).objects.all():
        if key.key.startswith('data/data_') and key.key < f'data/data_{two_month_ago}.json':
            s3.delete_file(key.key)
