import os

from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")


def train():
    print("Data ở", DATA_PATH)
    print("Chưa làm gì")
    pass


def predict(data):
    print("Trả ra giá trị dự đoán ở hàm này")
    pass


if __name__ == "__main__":
    train()
