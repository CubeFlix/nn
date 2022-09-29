// nn.go
// Main nn package code.

package nn

import (
	"log"
	"os"
	"io"
)


// Version.
const (
	VERSION = "1.0.3"
)


// Logging.
var (
	WarningLogger *log.Logger
	InfoLogger    *log.Logger
	ErrorLogger   *log.Logger
)


// Initialize the loggers.
func InitLogger(verbose, useFile bool, file string) error {
	// Initialize the logger objects.
	if useFile {
		// Use a log file.
		file, err := os.OpenFile(file, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0666)
		if err != nil {
			return err
		}

		InfoLogger = log.New(file, "[info]: ", log.Ldate|log.Ltime)
		WarningLogger = log.New(file, "[warning]: ", log.Ldate|log.Ltime)
		ErrorLogger = log.New(file, "[error]: ", log.Ldate|log.Ltime)
	} else {
		// Don't use a log file and instead write to stdout.
		var output io.Writer
		if verbose {
			// If verbose, output to stdout.
			output = os.Stdout
		} else {
			output = io.Discard
		}
		InfoLogger = log.New(output, "[info]: ", log.Ldate|log.Ltime)
                WarningLogger = log.New(output, "[warning]: ", log.Ldate|log.Ltime)
                ErrorLogger = log.New(output, "[error]: ", log.Ldate|log.Ltime)
	}

	return nil
}
