import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as Mdates
import matplotlib.cbook as cbook

def draw_chart(df):
    plt.figure()
    df.open.plot(legend=True)
    df.close.plot(legend=True)
    df.high.plot(legend=True)
    df.low.plot(legend=True)
    df.volume.plot(secondary_y=True,legend=True)

if __name__=='__main__':
    from preprocess import data_pre_pro_walk_pandas
    df = data_pre_pro_walk_pandas('2017data','FAX')
    draw_chart(df)


