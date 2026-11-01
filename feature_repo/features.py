from datetime import timedelta
from feast import FeatureView, Field, FileSource
from feast.types import Float32, Int64
from entities import account

source = FileSource(path="data/customer_features.parquet",timestamp_field="event_timestamp")

account_features = FeatureView(name="account_features",entities=[account],ttl=timedelta(days=7),
    schema=[Field(name="txn_count_24h", dtype=Int64),
        Field(name="transfer_count_24h", dtype=Int64),
        Field(name="cashout_count_24h", dtype=Int64),
        Field(name="avg_amount_24h", dtype=Float32),],source=source)
