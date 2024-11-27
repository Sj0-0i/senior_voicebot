### Prepare
* MySQL
* miniconda / anaconda
* API Keys
  * ChatGPT API KEY
  * OPENWEATHER API KEY

### Install

#### install python dependency
```shell
conda env create -f environment.yml
conda activate bytephant
```

#### create table

```sql
create database chatbot;

use chatbot;
    
create table questions
(
    question_id   int auto_increment
        primary key,
    question_text text null
);

create table users
(
    user_id   varchar(255) not null
        primary key,
    user_name varchar(255) null,
    age       int          null,
    location  varchar(255) null,
    constraint user_name
        unique (user_name)
);

create table userinterests
(
    interest_id int auto_increment
        primary key,
    interest    varchar(255)         not null,
    created_at  timestamp            null,
    user_id     varchar(255)         null,
    count       int        default 0 null,
    selected    tinyint(1) default 0 null,
    constraint unique_user_interest
        unique (user_id, interest),
    constraint user_interest_users__fk
        foreign key (user_id) references users (user_id)
);

create table userquestions
(
    user_id     varchar(255)                        not null,
    question_id int                                 not null,
    asked_at    timestamp default CURRENT_TIMESTAMP null,
    primary key (user_id, question_id),
    constraint userquestions_ibfk_1
        foreign key (user_id) references users (user_id),
    constraint userquestions_ibfk_2
        foreign key (question_id) references questions (question_id)
);

create index question_id
    on userquestions (question_id);

#하드 코딩된 유저 정보 입력
insert into users values ('abcdef','박호산',70,'Seoul');

# questions 테이블에 관심사 없을 때 말할 질문을 insert
insert into questions values (1,'오늘은 어떤 일을 할 계획인가요?');
insert into questions values (2,'쉴 떄 주로 하시는 일이 무엇인가요?');
```

#### .env 파일 작성
```shell
OPENAI_API_KEY=
OPENWEATHER_API_KEY=
MYSQL_USER=
MYSQL_PASSWORD=
MYSQL_HOST= # localhost or db domain
MYSQL_DB= # chatbot
```


### Run
```shell
uvicorn main:app --host 0.0.0.0
```