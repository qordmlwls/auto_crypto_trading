import boto3


class S3:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.s3 = boto3.resource('s3')
        
    def upload_file(self, file_name, key):
        """
        key: 저장될 파일명
        file_name: 업로드할 파일명
        """
        self.s3.Bucket(self.bucket_name).upload_file(file_name, key)
