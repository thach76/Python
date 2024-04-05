import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error


#load data
data = pd.read_csv(r"E:\Data\Download\Zalo Received Files\Fish.csv")

#chữ cái hoa X - dữ liệu ma trận - mảng 2 chiều
#     thường y - dữ liệu là vector - mảng 1 chiều
data.drop(['Species'], axis=1, inplace=True)    # bỏ cột Species
X = data.iloc[: , [1, 2, 3, 4, 5]]              # lấy dữ liệu từ các cột Length1,Length2,Length3,Height,Width
y = data.iloc[: , 0]                            # # lấy dữ liệu từ cột 0 là Weight

# Tách thành lập các tập Train (2/3) và Test (1/3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Huấn luyện mô hình hồi quy tuyến tính LinearRegression
LM = LinearRegression()         # Mô hình LinearRegression
LM.fit(X_train, y_train)        # Gọi thủ tục huấn luyện

# Hiệu suất mô hình
print('Hiệu suất huấn luyện:    ', LM.score(X_train, y_train))
print('Hiệu suất test:          ', LM.score(X_test, y_test))


# Dự đoán
y_pred = LM.predict(X_test)
print('Các giá trị Weight dự đoán là : \n')
print(y_pred)
print('-------------------------------------')

# Tính toán lỗi trong quá trình huấn luyện
# Calulating Mean Absolute Error MAE - Lỗi tuyệt đối trung bình
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
print('Mean Absolute Error Value MAE: ', MAEValue)

# Calulating Mean squared Error MSE - Lỗi bình phương trung bình
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
print('Mean squared Error Value MSE: ', MSEValue)

# Calulating Median squared Error  - Lỗi bình phương trung điểm
MDsEValue = median_absolute_error(y_test, y_pred)
print('Median squared Error Value MdSE: ', MDsEValue)

# Đồ thị
plt.xlabel('Các đặc trưng')
plt.ylabel('Trọng lượng')
plt.title('Dự đoán trọng lượng cá')
plt.plot(X_test, y_pred, "ro", label='Dự đoán')
plt.plot(X_train, y_train, "b^", label='Huấn luyện')
plt.legend()    #Hộp chú thích
plt.show()

print('Done!')