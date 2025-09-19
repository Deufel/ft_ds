-- Migration 002: Add priority levels and due dates to tasks
-- This migration adds priority and due_date columns to enhance task management

-- Add priority column with default 'medium'
ALTER TABLE tasks ADD COLUMN priority TEXT DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'urgent'));

-- Add due_date column (nullable - not all tasks need due dates)
ALTER TABLE tasks ADD COLUMN due_date TEXT DEFAULT NULL;

-- Add updated_at column to track when tasks are modified
ALTER TABLE tasks ADD COLUMN updated_at TEXT DEFAULT (datetime('now'));

-- Create an index on priority for faster filtering
CREATE INDEX idx_tasks_priority ON tasks(priority);

-- Create an index on due_date for deadline queries
CREATE INDEX idx_tasks_due_date ON tasks(due_date);

-- Create a composite index for common queries (completed + priority)
CREATE INDEX idx_tasks_completed_priority ON tasks(completed, priority);

-- Update existing tasks to have an updated_at timestamp
UPDATE tasks SET updated_at = created_at WHERE updated_at IS NULL;

-- Add some sample data with the new columns
INSERT INTO tasks (title, description, priority, due_date) VALUES 
    ('Fix critical bug', 'Resolve the authentication issue in production', 'urgent', date('now', '+1 day')),
    ('Write documentation', 'Document the new API endpoints', 'low', date('now', '+1 week')),
    ('Code review', 'Review pull requests from team members', 'high', date('now', '+2 days')),
    ('Plan sprint', 'Organize tasks for next development sprint', 'medium', date('now', '+3 days'));

-- Create a view for overdue tasks (optional but useful)
CREATE VIEW overdue_tasks AS
SELECT 
    task_id,
    title,
    description,
    priority,
    due_date,
    created_at,
    updated_at,
    julianday(due_date) - julianday('now') as days_overdue
FROM tasks 
WHERE completed = 0 
  AND due_date IS NOT NULL 
  AND due_date < date('now')
ORDER BY priority DESC, due_date ASC;