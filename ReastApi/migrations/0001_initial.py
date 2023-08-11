# Generated by Django 4.1 on 2023-08-09 15:48

from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Api',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False, unique=True)),
                ('api', models.TextField()),
                ('use', models.IntegerField(default=0)),
            ],
        ),
        migrations.CreateModel(
            name='CsrfTokenNew',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Ip', models.GenericIPAddressField()),
                ('Hash', models.CharField(max_length=65)),
                ('UserAgent', models.TextField()),
                ('Time', models.DateTimeField(default=django.utils.timezone.now)),
            ],
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False, unique=True)),
                ('username', models.CharField(max_length=20, unique=True)),
                ('password', models.CharField(max_length=20)),
                ('email', models.EmailField(max_length=254, unique=True)),
            ],
        ),
        migrations.CreateModel(
            name='Sentiment',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False, unique=True)),
                ('text', models.TextField()),
                ('sentiment', models.CharField(max_length=20)),
                ('code', models.CharField(max_length=40)),
                ('api', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ReastApi.api')),
            ],
        ),
        migrations.CreateModel(
            name='HashUserApi',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False, unique=True)),
                ('Hash', models.TextField()),
                ('DataCreate', models.DateTimeField(auto_now_add=True)),
                ('Ip', models.CharField(max_length=15)),
                ('UserAgent', models.TextField()),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ReastApi.user')),
            ],
        ),
        migrations.AddField(
            model_name='api',
            name='user_content',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ReastApi.user'),
        ),
        migrations.AddField(
            model_name='api',
            name='user_login',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ReastApi.hashuserapi'),
        ),
    ]