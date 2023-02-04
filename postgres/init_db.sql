CREATE TABLE IF NOT EXISTS action_logs (
    id bigserial primary key,
    action_type varchar(30) NOT NULL,
    action_datetime TIMESTAMP,
    code int NOT NULL,
    error_message varchar(250)
);