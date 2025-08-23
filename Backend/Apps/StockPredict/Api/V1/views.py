from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from ...Services import DataService, PredictionService

class StockViewSet(viewsets.ViewSet):
    """
    API endpoint for stock predictions
    """

    @action(detail=False, methods=['post'])
    def predict(self, request):
        ticker = request.data.get('ticker')
        timeframe = request.data.get('timeframe', '1d')

        # Use services
        data_service = DataService()
        prediction_service = PredictionService()

        # Process request
        data = data_service.fetch_stock_data(ticker, timeframe)
        prediction = prediction_service.make_prediction(data)

        return Response(prediction)
