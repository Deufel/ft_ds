CREATE TABLE tasks (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    completed INTEGER DEFAULT 0 CHECK (completed IN (0, 1)),
    created_at TEXT DEFAULT (datetime('now'))
);

INSERT INTO tasks (title, description) VALUES 
    ('Complete project setup', 'Set up database and migrations'),
    ('Build task management', 'Create CRUD operations for tasks'),
    ('Add bulk operations', 'Implement bulk complete/incomplete');