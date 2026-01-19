import pandas as pd
import os

# 获取当前目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir= r'D:\yuhan\deepbgc-master'
base_dir = r'D:\yuhan\deepbgc-master'
folder_names = [
    'arabidopsis_cds_from_genomic','brassica_oleracea_cds_from_genomic', 'capsella cds_from_genomic', 'ccucumis sativus_ds_from_genomic',
    'hordeumcds_from_genomic', 'jatropha_curcas_cds_from_genomic', 'lotus_japonicus_cds_from_genomic', 'nicotiana_tabacum_cds_from_genomic',
    'oryza_stiva_cds_from_genomic','Papaver_somniferumcds_from_genomic', 'Ricinus_communis_cds_from_genomic', 'salvia_splendens_cds_from_genomic',
    'solanum_lycopersicum_cds_from_genomic', 'solanum_tubersum_cds_from_genomic','sorghun_bicolor_cds_from_genomic','spinacia_oleracea_cds_from_genomic',
    'Taxus_chinensis_cds_from_genomic','zea_mays_cds_from_genomic','Sorghum_bicolor.Sorghum_bicolor_NCBIv3.cdna.all'
]

folder_name='Sorghum_bicolor.Sorghum_bicolor_NCBIv3.cdna.all'
# 定义输入文件路径
# bgc_prediction_path = os.path.join(current_dir, '../arabidopsis_cds_from_genomic/arabidopsis_cds_from_genomic.bgc.tsv')

bgc_prediction_path = os.path.join(current_dir,folder_name, 'Sorghum_bicolor.Sorghum_bicolor_NCBIv3.cdna.all.bgc.tsv')
# 读取DeepBGC预测得到的BGC片段的TSV文件
bgc_prediction_df = pd.read_csv(bgc_prediction_path, sep='\t')
print(f"bgc_prediction_path: {bgc_prediction_path}")
# 初始化合并的列表
merged_rows = []

# 用于跟踪当前需要合并的sequence_id和对应的其他列
current_sequence_ids = []
current_row = None

# 遍历每一行
for index, row in bgc_prediction_df.iterrows():
    if current_row is None:
        # 第一次进入循环时，初始化current_row
        current_row = row
        current_sequence_ids.append(row['sequence_id'])
    else:
        # 如果pfam_ids相同，合并sequence_id
        if row['pfam_ids'] == current_row['pfam_ids']:
            current_sequence_ids.append(row['sequence_id'])
        else:
            # 如果pfam_ids不同，将当前的sequence_id合并，并添加到merged_rows
            current_row['sequence_id'] = ','.join(current_sequence_ids)
            merged_rows.append(current_row)

            # 更新current_row为新的行
            current_row = row
            current_sequence_ids = [row['sequence_id']]

# 添加最后一行合并的结果
if current_row is not None:
    current_row['sequence_id'] = ','.join(current_sequence_ids)
    merged_rows.append(current_row)

# 将合并后的数据转换为DataFrame
merged_df = pd.DataFrame(merged_rows)

# 定义输出路径
output_dir = os.path.join(current_dir, folder_name)
output_path = os.path.join(output_dir, 'merged_bgc_output_with_pfam.tsv')

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存结果为tsv文件
merged_df.to_csv(output_path, sep='\t', index=False)

print(f"合并完成，文件已保存至: {output_path}")
