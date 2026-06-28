import pandas as pd 
import numpy as np 
import pybaseball 
import time 

statcast_drop_cols = [
    'spin_dir',           
    'spin_rate_deprecated',
    'break_angle_deprecated',
    'break_length_deprecated',
    'tfs_deprecated',
    'tfs_zulu_deprecated',
    'umpire',               
    'sv_id',                
    'hit_distance_sc',     
    'launch_speed',         
    'launch_angle',         
    'hc_x', 'hc_y',        
    'estimated_ba_using_speedangle',   
    'estimated_woba_using_speedangle', 
    'woba_value',          
    'woba_denom',
    'babip_value',
    'iso_value',
    'delta_home_win_exp',   
    'delta_run_exp',  
    'home_score', 
    'away_score', 
    'bat_score', 
    'fld_score', 
    'post_away_score', 
    'post_home_score', 
    'post_bat_score', 
    'post_fld_score',
    'home_score_diff', 
    'bat_score_diff', 
    'home_win_exp', 
    'bat_win_exp',
    'if_fielding_alignment', 
    'of_fielding_alignment',
    'fielder_2',
    'fielder_3',
    'fielder_4',
    'fielder_5',
    'fielder_6',
    'fielder_7',
    'fielder_8',
    'fielder_9','launch_speed_angle', 
    'estimated_slg_using_speedangle', 
    'age_bat', 'age_bat_legacy', 
    'batter_days_since_prev_game',
    'batter_days_until_next_game', 'attack_angle',
    'bat_speed', 'swing_length', 
    'swing_path_tilt', 'attack_direction', 
    'intercept_ball_minus_batter_pos_x_inches', 
    'intercept_ball_minus_batter_pos_y_inches'
]

##pull by month to avoid crash
chunks = []
ranges = [
         ('2021-04-01','2021-04-30'),('2021-05-01','2021-05-31'),
               ('2021-06-01','2021-06-30'),('2021-07-01','2021-07-31'),
               ('2021-08-01','2021-08-31'),('2021-09-01','2021-10-03'),
        ('2022-04-07','2022-04-30'),('2022-05-01','2022-05-31'),
               ('2022-06-01','2022-06-30'),('2022-07-01','2022-07-31'),
               ('2022-08-01','2022-08-31'),('2022-09-01','2022-10-05'),
        ('2023-03-30','2023-04-30'),('2023-05-01','2023-05-31'),
               ('2023-06-01','2023-06-30'),('2023-07-01','2023-07-31'),
               ('2023-08-01','2023-08-31'),('2023-09-01','2023-10-01'),
        ('2024-03-20','2024-04-30'),('2024-05-01','2024-05-31'),
               ('2024-06-01','2024-06-30'),('2024-07-01','2024-07-31'),
               ('2024-08-01','2024-08-31'),('2024-09-01','2024-09-29'),
    ]


for start, end in ranges:
    time.sleep(5)
    chunk = pybaseball.statcast(start_dt=start, end_dt=end)
    if chunk is not None:
        chunk=chunk.drop(columns=statcast_drop_cols)
        chunks.append(chunk)

statcast_pbp_df = pd.concat(chunks, ignore_index=True)
statcast_pbp_df = statcast_pbp_df.rename(columns={'pitcher': 'mlbam_id'})

statcast_pbp_df.to_csv('statcast_pbp.csv')
##possibly try to filter out position player appearnaces later

statcast_pbp=pd.read_csv('/Users/tommygillan/Documents/Unstructured/statcast_pbp.csv')
fastball_variations  = {'FF', 'SI', 'FC'}

statcast_pbp['is_fastball'] = statcast_pbp['pitch_type'].isin(fastball_variations)
statcast_pbp['is_breaking'] = (~statcast_pbp['pitch_type'].isin(fastball_variations))
statcast_pbp['game_date'] = pd.to_datetime(statcast_pbp_df['game_date'])
statcast_pbp['season'] = statcast_pbp['game_date'].dt.year

game_level_df = (
        statcast_pbp
        .groupby(['mlbam_id', 'game_pk', 'game_date', 'season'])
        .agg(

            pitch_count_game     = ('release_speed', 'count'),
            avg_fb_velo_game     = ('release_speed',
                                    lambda x: x[statcast_pbp.loc[x.index,'is_fastball']].mean()),
            avg_spin_game        = ('release_spin_rate', 'mean'),
            avg_rel_x_game       = ('release_pos_x', 'mean'),
            avg_rel_z_game       = ('release_pos_z', 'mean'),
            std_rel_x_game       = ('release_pos_x', 'std'),
            std_rel_z_game       = ('release_pos_z', 'std'),
            avg_extension_game   = ('release_extension', 'mean'),
            avg_pfx_x_game       = ('pfx_x', 'mean'),
            avg_pfx_z_game       = ('pfx_z', 'mean'), 
            breaking_pct_game    = ('is_breaking', 'mean'),
            max_inning_game      = ('inning', 'max'),
            pitch_hand           = ('p_throws', 'first'),
        )
        .reset_index()
        .sort_values(['mlbam_id', 'game_date'])
    )
def add_rolling(df):
    df = df.sort_values(['mlbam_id', 'game_date']).reset_index(drop=True)
    grouped = df.groupby('mlbam_id')
    for days in [7, 15, 28, 60]:
        df[f'pitch_count_{days}d'] = (
            grouped.rolling(f'{days}D', on='game_date', closed='left')['pitch_count_game']
            .sum()
            .reset_index( drop=True)
            .fillna(0))
    ##99 rest days = first appearance of the year
    df['days_rest'] = grouped['game_date'].diff().dt.days.fillna(99)
    df['velo_15d_avg'] = (
        grouped.rolling('15D', on='game_date', closed='left')['avg_fb_velo_game']
        .mean()
        .reset_index( drop=True)
    )
    
    df['velo_60d_avg'] = (
        grouped.rolling('60D', on='game_date', closed='left')['avg_fb_velo_game']
        .mean()
        .reset_index(drop=True)
    )
  
    df['rel_z_drift_15d'] = (
        grouped.rolling('15D', on='game_date', closed='left')['std_rel_z_game']
        .mean()
        .reset_index(drop=True)
    )
    df['rel_x_drift_15d'] = (
        grouped.rolling('15D', on='game_date', closed='left')['std_rel_x_game']
        .mean()
        .reset_index(drop=True)
    )


    df['max_pitches_30d'] = (
        grouped.rolling('30D', on='game_date', closed='left')['pitch_count_game']
        .max()
        .reset_index(drop=True)
        .fillna(0)
    )

    df['spin_rate_7d_avg'] = (
        grouped.rolling('15D', on='game_date', closed='left')['avg_spin_game']
        .mean()
        .reset_index(drop=True)
    )


    df['extension_15d_avg'] = (
        grouped.rolling('15D', on='game_date', closed='left')['avg_extension_game']
        .mean()
        .reset_index(drop=True)
    )

    df['vert_break_15d_avg'] = (
        grouped.rolling('15D', on='game_date', closed='left')['avg_pfx_z_game']
        .mean()
        .reset_index(drop=True)
    )
    df['spin_rate_28d_avg'] = (
        grouped.rolling('28D', on='game_date', closed='left')['avg_spin_game']
        .mean()
        .reset_index(drop=True)
    )

    df['acwr_pitch_count'] = df['pitch_count_7d'] / ((df['pitch_count_28d']/4) + 1)
    df['acwr_velo'] = df['velo_15d_avg'] / (df['velo_60d_avg'] + 1e-6)
    df['acwr_spin'] = df['spin_rate_7d_avg'] / (df['spin_rate_28d_avg'] + 1e-6)
    return df

game_level_df = add_rolling(game_level_df)


lahman= pd.read_csv("/Users/tommygillan/Downloads/lahman_1871-2024_csv/People.csv")

info = lahman[['retroID','weight','height','birthYear','birthMonth','birthDay']].copy()
info=info.dropna(subset=['retroID'])
##all different player IDs that exist
chadwick = pybaseball.chadwick_register()
bio = info.merge(
    chadwick,
    left_on='retroID', right_on='key_retro',
    how='inner'
).rename(columns={'key_mlbam': 'mlbam_id'})


import statsapi


def pull_il(seasons, pitcher_ids):
    arm_keywords = [
        'elbow', 'ucl', 'ulnar', 'tommy john', 'flexor',
        'shoulder', 'rotator', 'labrum',
        'forearm', 'bicep', 'biceps', 'tricep',
        'finger', 'hand', 'wrist', 'blister', 'nerve']
      
    teams = statsapi.get('teams', {'sportId': 1})
    team_ids = [team['id'] for team in teams['teams']]
    all_records = []
    
    for season in seasons:
        print(f"Processing Season: {season}")
        for team_id in team_ids:
            params = {
                'teamId': team_id,
                'startDate': f'{season}-03-01',
                'endDate': f'{season}-11-30'
            }
            raw = statsapi.get('transactions', params)
            transactions = raw.get('transactions', [])
                
            for t in transactions:
                desc = t.get('description', '').lower()
                p_id = t.get('person', {}).get('id')
                    
                    
                if 'placed' in desc and 'injured list' in desc:
                    if p_id in pitcher_ids:
                        if any(keyword in desc for keyword in arm_keywords):
                            all_records.append({
                                'mlbam_id': p_id,
                                'player_name': t.get('person', {}).get('fullName'),
                                'il_start_date': t.get('date'),
                                'description': t.get('description')
                            })
            
              
    df = pd.DataFrame(all_records)
    if not df.empty:
        df['il_start_date'] = pd.to_datetime(df['il_start_date']).dt.normalize()
        df = df.drop_duplicates(subset=['mlbam_id', 'il_start_date'])
        
    return df


unique_pitchers = set(game_level_df['mlbam_id'].unique())
il_df = pull_il([2021, 2022, 2023, 2024], unique_pitchers)


game_level_df['game_date'] = pd.to_datetime(game_level_df['game_date'])
il_df['il_start_date'] = pd.to_datetime(il_df['il_start_date'])
game_level_df = game_level_df.dropna(subset=['game_date'])
il_df = il_df.dropna(subset=['il_start_date'])
game_level_df = game_level_df.sort_values('game_date').reset_index(drop=True)
il_df = il_df.sort_values( 'il_start_date').reset_index(drop=True)


merged_gl_df = pd.merge_asof(
        left=game_level_df,
        right=il_df[['mlbam_id', 'il_start_date', 'description']], 
        by='mlbam_id',
        left_on='game_date',
        right_on='il_start_date',
        direction='forward' 
    )
merged_gl_df['days_to_injury'] = (merged_gl_df['il_start_date'] - merged_gl_df['game_date']).dt.days
merged_gl_df['target_inj_14d'] = ((merged_gl_df['days_to_injury'] > 0) & (merged_gl_df['days_to_injury'] <= 14)).astype(int)


bio = bio.dropna(subset=['mlbam_id'])
bio['mlbam_id'] = bio['mlbam_id'].astype(int)
merged_gl_df['mlbam_id'] = merged_gl_df['mlbam_id'].astype(int)


lehman_columns = ['mlbam_id', 'weight', 'height', 'birthYear']

merged_gl_df = merged_gl_df.merge(
    bio[lehman_columns], 
    on='mlbam_id', 
    how='left'
)

merged_gl_df['pitcher_age'] = merged_gl_df['season'] - merged_gl_df['birthYear']
merged_gl_df.to_csv('final_gl_df.csv', index=False)



merged_gl_df['target_inj_14d'].value_counts()
merged_gl_df['days_to_injury'].describe()
merged_gl_df['max_pitches_30d'].describe()
merged_gl_df['avg_fb_velo_game'].describe()
merged_gl_df['breaking_pct_game'].describe()
 
from plotnine import*

dense=(
    ggplot(merged_gl_df)
    +aes(x='pitcher_age', fill='factor(target_inj_14d)')
    + geom_density(alpha=0.5, bw=1.5)
         + theme(                                             
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_blank(),
        panel_background=element_blank()
    )
    + labs(title="Pitcher Age Density Plot",x='Pitcher Age', fill="Healthy Vs Injured")
)
dense


injured_subset=merged_gl_df[merged_gl_df['target_inj_14d']==1]

bar=(
    ggplot(injured_subset)
    +aes(x= 'season')
    +geom_bar(fill='green')
      + theme(                                             
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_blank(),
        panel_background=element_blank()
    )
    + labs(title="Injuries Per Season",x='Season',y='Total_injuries')
)
bar

subset=merged_gl_df[merged_gl_df['pitch_count_game']>=18]
histogram=(
    ggplot(subset)
    +aes(x='avg_fb_velo_game')
    +geom_histogram()
    +theme(                                             
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_blank(),
        panel_background=element_blank()
    )
    + labs(title="AVG Fastball Velocity Histogram",x='AVG Fastball Velocity')
    +facet_wrap('season')
)
histogram

merged_gl_df['acwr_pitch_count'].describe()
merged_gl_df['acwr_velo'].describe()
merged_gl_df['acwr_spin'].describe()
merged_gl_df['height'].describe()
merged_gl_df['weight'].describe()
merged_gl_df['age'].describe()
merged_gl_df['max_pitches_30d'].describe()
