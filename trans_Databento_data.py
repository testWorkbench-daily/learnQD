import databento as db
import pandas as pd

# 2. 读取下载的文件
data = db.DBNStore.from_file('GLBX-20251220-UVDVWP76FX/glbx-mdp3-20100606-20251219.ohlcv-1m.dbn.zst')

# 3. 查看数据结构
df = data.to_df()
print(df.head())
print(df.columns)

# 4. 存成 CSV
df.to_csv('nq_m1_all.csv')

# 5. 处理成 Backtrader 格式
df.index = pd.to_datetime(df.index).tz_localize(None)
df = df[['open', 'high', 'low', 'close', 'volume']]
df.to_csv('nq_m1_all_backtrader.csv')