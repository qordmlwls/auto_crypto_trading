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
    
    def download_file(self, key, file_name):
        self.s3.Bucket(self.bucket_name).download_file(key, file_name)

    def delete_file(self, key):
        self.s3.Object(self.bucket_name, key).delete()
        