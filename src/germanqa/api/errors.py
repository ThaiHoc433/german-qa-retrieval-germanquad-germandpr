"""Lỗi của API (GĐ8): một dạng JSON duy nhất {"error": {code, message, request_id, details?}}."""


class ApiError(Exception):
    def __init__(self, status, code, message, details=None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


def error_body(code, message, request_id=None, details=None):
    body = {"code": code, "message": message, "request_id": request_id}
    if details:
        body["details"] = details
    return {"error": body}
