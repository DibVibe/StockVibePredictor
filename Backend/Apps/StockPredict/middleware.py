from django.http import JsonResponse

class APIVersionMiddleware:
    """Middleware to handle API versioning and deprecation warnings"""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Add version info to headers
        response = self.get_response(request)

        # Add API version headers
        response['X-API-Version'] = '2.1.0'
        response['X-API-Versions-Supported'] = 'v1'

        # Add deprecation warnings if using old endpoints
        if '/api/' in request.path and '/api/v1/' not in request.path:
            response['X-API-Deprecation-Warning'] = 'Please use /api/v1/ for future compatibility'

        return response
