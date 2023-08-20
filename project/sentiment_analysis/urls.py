from django.urls import path
from . import views

urlpatterns = [
    path('analyze/', views.analyze_sentiment, name='analyze_sentiment'),
    path('analyze1/', views.analyze_sentiment1, name='analyze_sentiment'),
]
