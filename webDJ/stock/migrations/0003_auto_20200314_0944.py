# Generated by Django 3.0.3 on 2020-03-14 04:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stock', '0002_remove_stock_price'),
    ]

    operations = [
        migrations.CreateModel(
            name='About',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Company_name', models.TextField()),
                ('price', models.IntegerField()),
            ],
        ),
        migrations.DeleteModel(
            name='Stock',
        ),
    ]
