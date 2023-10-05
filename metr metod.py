# Импортируем необходимые библиотеки
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Загрузим данные о цветках ириса
iris = load_iris()
X = iris.data
y = iris.target

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создадим объект классификатора KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Обучим классификатор на обучающей выборке
knn.fit(X_train, y_train)

# Сделаем предсказания на тестовой выборке
y_pred = knn.predict(X_test)

# Оценим точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy}')