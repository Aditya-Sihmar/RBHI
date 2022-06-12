from rest_framework import serializers
from .models import Class1

class CourseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Class1
        fields = ('id', 'name', 'language','price')