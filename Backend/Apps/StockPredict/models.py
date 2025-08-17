from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class PredictionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    ticker = models.CharField(max_length=10)
    timeframe = models.CharField(max_length=10)
    prediction = models.JSONField()
    actual_outcome = models.BooleanField(null=True, blank=True)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    verified_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['ticker', 'timeframe']),
            models.Index(fields=['created_at']),
        ]

class ModelPerformance(models.Model):
    model_name = models.CharField(max_length=100)
    ticker = models.CharField(max_length=10, null=True, blank=True)
    timeframe = models.CharField(max_length=10)
    accuracy = models.FloatField()
    precision = models.FloatField(null=True)
    recall = models.FloatField(null=True)
    f1_score = models.FloatField(null=True)
    training_samples = models.IntegerField(null=True)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['model_name', 'ticker', 'timeframe']

class ApiUsage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    endpoint = models.CharField(max_length=100)
    ticker = models.CharField(max_length=10, null=True, blank=True)
    response_time = models.FloatField()
    status_code = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
