# HSE_MLOps_Api_Suleev

#### Введение

Приветсвую Глеба Борисенко и других случайно забредших.
Здесь расскажу о имеющихся в проекте функциях и о том, как они работают.

## First_HW

### 1. fit_model
Функция, обучающая модели. 

Обязательно нужно задать тип обучаемой модели **model_type**, 
данные для обучения **train_data** и таргет **train_target**. 

Можно задать название модели **model_name** самому, но если данный ключ 
не задавать, то название подберется автоматически. Нельзя подавать
уже существующие названия моделей.

Гиперпараметры **hyperparams** подаются в виде словаря. Если не указывать
данный ключ, то будут взяты параметры по умолчанию

### 2. refit_model
Переобучает существующую модель.

Обязательно подаем название существующей модели **model_name**
и данные с таргетом **train_data** и **train_target**.

При необходимости, также можно задать свои гиперпараметры **hyperparams** 

### 3. remove_model
Удаляет существующую модель.

По ключу **model_names** в списке перечисляем все модели,
которые хотим удалить

### 4. predict
Предсказыват вероятность наблюдения/наблюдений пренадлежать 
к классу 1.

Необходимо подать имя модели **"model_name"** и данные **"data"**.

Также можно задать **cut_off**, чтобы возвращался 
предсказанный класс наблюдений, а не просто вероятность

### 5. show

Выводит список всех существующих моделей или параметры
одной обученной модели.

Работа осуществляется по ключу **model_name**. Если значение 
равно **All**, то будет выведен список доступных моделей.
Если же в качестве значения подать имя модели, то выведутся
её гиперпараметры


## Second_HW

Образ лежит на DockerHub под названием dsuleev/hse_mlops_api_suleev

Логин: dsuleev

### Запуск образа

```bash
docker compose build
docker compose up -d
```

### Работа с БД

В проекте логируется взаимодействие с API в формате 
**название действия**, **дата и время действия**, 
**http-код**, **текст ошибки** (если запрос был завершен с ошибкой)

Посмотреть на таблицу можно с помощью следущих команд

```bash
docker exec -it hse_mlops_api_suleev-db-1 /bin/bash
psql -U dsuleev mlops_api

SELECT * FROM action_logs;
```

## Third_HW

Фикстуры находятся в файле `confest.py`

Для запуска тестов нужно ввести команду 
```bash
pytest -s -v unit_tests/*
```

(Хотел воспользоваться параметризацией для теста функций 
`fit_model` и `refit_model`, однако набор обязательных 
параметров там отличается)