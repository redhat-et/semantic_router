package extproc

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"
)

// LogLevel represents the severity level of a log message
type LogLevel int

const (
	LogLevelDebug LogLevel = iota
	LogLevelInfo
	LogLevelWarn
	LogLevelError
)

// String returns the string representation of the log level
func (l LogLevel) String() string {
	switch l {
	case LogLevelDebug:
		return "DEBUG"
	case LogLevelInfo:
		return "INFO"
	case LogLevelWarn:
		return "WARN"
	case LogLevelError:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

// StructuredLogger implements the Logger interface with structured logging
type StructuredLogger struct {
	level      LogLevel
	component  string
	jsonFormat bool
	logger     *log.Logger
}

// NewStructuredLogger creates a new structured logger
func NewStructuredLogger(component string, level LogLevel, jsonFormat bool) *StructuredLogger {
	return &StructuredLogger{
		level:      level,
		component:  component,
		jsonFormat: jsonFormat,
		logger:     log.New(os.Stdout, "", 0), // No prefix, we'll format ourselves
	}
}

// LogEntry represents a structured log entry
type LogEntry struct {
	Timestamp string                 `json:"timestamp"`
	Level     string                 `json:"level"`
	Component string                 `json:"component"`
	Message   string                 `json:"message"`
	Error     string                 `json:"error,omitempty"`
	Fields    map[string]interface{} `json:"fields,omitempty"`
}

// Info logs an info message
func (l *StructuredLogger) Info(msg string, fields ...Field) {
	if l.level <= LogLevelInfo {
		l.log(LogLevelInfo, msg, nil, fields...)
	}
}

// Error logs an error message
func (l *StructuredLogger) Error(msg string, err error, fields ...Field) {
	if l.level <= LogLevelError {
		l.log(LogLevelError, msg, err, fields...)
	}
}

// Debug logs a debug message
func (l *StructuredLogger) Debug(msg string, fields ...Field) {
	if l.level <= LogLevelDebug {
		l.log(LogLevelDebug, msg, nil, fields...)
	}
}

// Warn logs a warning message
func (l *StructuredLogger) Warn(msg string, fields ...Field) {
	if l.level <= LogLevelWarn {
		l.log(LogLevelWarn, msg, nil, fields...)
	}
}

// log is the internal logging method
func (l *StructuredLogger) log(level LogLevel, msg string, err error, fields ...Field) {
	entry := LogEntry{
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Level:     level.String(),
		Component: l.component,
		Message:   msg,
		Fields:    make(map[string]interface{}),
	}

	if err != nil {
		entry.Error = err.Error()
	}

	// Add fields
	for _, field := range fields {
		entry.Fields[field.Key] = field.Value
	}

	if l.jsonFormat {
		l.logJSON(entry)
	} else {
		l.logText(entry)
	}
}

// logJSON outputs the log entry as JSON
func (l *StructuredLogger) logJSON(entry LogEntry) {
	if data, err := json.Marshal(entry); err == nil {
		l.logger.Println(string(data))
	} else {
		// Fallback to simple text if JSON marshaling fails
		l.logger.Printf("[%s] %s: %s", entry.Level, entry.Component, entry.Message)
	}
}

// logText outputs the log entry as formatted text
func (l *StructuredLogger) logText(entry LogEntry) {
	var output string
	
	// Basic format: [LEVEL] component: message
	output = fmt.Sprintf("[%s] %s: %s", entry.Level, entry.Component, entry.Message)
	
	// Add error if present
	if entry.Error != "" {
		output += fmt.Sprintf(" error=%s", entry.Error)
	}
	
	// Add fields
	for key, value := range entry.Fields {
		output += fmt.Sprintf(" %s=%v", key, value)
	}
	
	l.logger.Println(output)
}

// SetLevel changes the log level
func (l *StructuredLogger) SetLevel(level LogLevel) {
	l.level = level
}

// GetLevel returns the current log level
func (l *StructuredLogger) GetLevel() LogLevel {
	return l.level
}

// WithFields creates a new logger with additional context fields
func (l *StructuredLogger) WithFields(fields ...Field) Logger {
	return &contextLogger{
		base:   l,
		fields: fields,
	}
}

// contextLogger wraps a logger with additional context fields
type contextLogger struct {
	base   *StructuredLogger
	fields []Field
}

func (c *contextLogger) Info(msg string, fields ...Field) {
	allFields := append(c.fields, fields...)
	c.base.Info(msg, allFields...)
}

func (c *contextLogger) Error(msg string, err error, fields ...Field) {
	allFields := append(c.fields, fields...)
	c.base.Error(msg, err, allFields...)
}

func (c *contextLogger) Debug(msg string, fields ...Field) {
	allFields := append(c.fields, fields...)
	c.base.Debug(msg, allFields...)
}

func (c *contextLogger) Warn(msg string, fields ...Field) {
	allFields := append(c.fields, fields...)
	c.base.Warn(msg, allFields...)
}

// NullLogger is a logger that does nothing (useful for testing)
type NullLogger struct{}

func (n *NullLogger) Info(msg string, fields ...Field)                {}
func (n *NullLogger) Error(msg string, err error, fields ...Field)    {}
func (n *NullLogger) Debug(msg string, fields ...Field)               {}
func (n *NullLogger) Warn(msg string, fields ...Field)                {}

// NewNullLogger creates a logger that discards all log messages
func NewNullLogger() Logger {
	return &NullLogger{}
}

// LoggerConfig represents logger configuration
type LoggerConfig struct {
	Level      string `yaml:"level"`
	Component  string `yaml:"component"`
	JSONFormat bool   `yaml:"json_format"`
}

// NewLoggerFromConfig creates a logger from configuration
func NewLoggerFromConfig(config LoggerConfig) Logger {
	level := parseLogLevel(config.Level)
	component := config.Component
	if component == "" {
		component = "extproc"
	}
	
	return NewStructuredLogger(component, level, config.JSONFormat)
}

// parseLogLevel parses a string log level
func parseLogLevel(levelStr string) LogLevel {
	switch levelStr {
	case "debug", "DEBUG":
		return LogLevelDebug
	case "info", "INFO":
		return LogLevelInfo
	case "warn", "WARN", "warning", "WARNING":
		return LogLevelWarn
	case "error", "ERROR":
		return LogLevelError
	default:
		return LogLevelInfo // Default to info
	}
}

// Helper functions for common logging patterns

// LogRequest logs a request processing event
func LogRequest(logger Logger, requestID, stage string, duration time.Duration, fields ...Field) {
	allFields := []Field{
		{Key: LogFieldRequestID, Value: requestID},
		{Key: LogFieldStage, Value: stage},
		{Key: LogFieldDuration, Value: duration.Milliseconds()},
	}
	allFields = append(allFields, fields...)
	logger.Info("Request processed", allFields...)
}

// LogError logs an error with standard fields
func LogError(logger Logger, requestID, component, operation string, err error, fields ...Field) {
	allFields := []Field{
		{Key: LogFieldRequestID, Value: requestID},
		{Key: LogFieldComponent, Value: component},
		{Key: LogFieldOperation, Value: operation},
	}
	allFields = append(allFields, fields...)
	logger.Error("Operation failed", err, allFields...)
}

// LogPlugin logs a plugin event
func LogPlugin(logger Logger, pluginName, operation string, fields ...Field) {
	allFields := []Field{
		{Key: LogFieldPlugin, Value: pluginName},
		{Key: LogFieldOperation, Value: operation},
	}
	allFields = append(allFields, fields...)
	logger.Info("Plugin operation", allFields...)
}

// LogSecurity logs a security event
func LogSecurity(logger Logger, requestID, eventType string, blocked bool, fields ...Field) {
	allFields := []Field{
		{Key: LogFieldRequestID, Value: requestID},
		{Key: "event_type", Value: eventType},
		{Key: "blocked", Value: blocked},
	}
	allFields = append(allFields, fields...)
	
	if blocked {
		logger.Warn("Security event blocked", allFields...)
	} else {
		logger.Debug("Security event allowed", allFields...)
	}
}

// LogCache logs a cache event
func LogCache(logger Logger, requestID string, hit bool, operation string, fields ...Field) {
	allFields := []Field{
		{Key: LogFieldRequestID, Value: requestID},
		{Key: LogFieldCacheHit, Value: hit},
		{Key: LogFieldOperation, Value: operation},
	}
	allFields = append(allFields, fields...)
	
	if hit {
		logger.Info("Cache hit", allFields...)
	} else {
		logger.Debug("Cache miss", allFields...)
	}
}