import React from "react";

const ErrorMessage = ({ error }) => {
  const isBackendError = error.includes("backend");

  return (
    <div className="error-message">
      <span className="error-icon">⚠️</span>
      <div className="error-content">
        <strong>Error:</strong> {error}
        {isBackendError && (
          <div className="error-help">
            <p>
              💡 <strong>Quick Fix:</strong>
            </p>
            <p>
              1. Make sure Django is running:{" "}
              <code>python manage.py runserver</code>
            </p>
            <p>
              2. Train models if needed: <code>python TrainModel.py full</code>
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ErrorMessage;
