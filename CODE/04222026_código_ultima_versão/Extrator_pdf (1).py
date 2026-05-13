import fitz  # PyMuPDF
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_alpha_vs_embed(csv_path,title):
    df = pd.read_csv(csv_path)

    # =========================
    # Dividir em 6 blocos iguais
    # =========================
    n = len(df) // 6

    blocks = [df.iloc[i*n:(i+1)*n] for i in range(6)]

    labels = [
        'M5 4x', 'M5 5x', 'M5 6x',
        'H1 4x', 'H1 5x', 'H1 6x'
    ]

    # =========================
    # Plot
    # =========================
    plt.figure(figsize=(7, 5))

    for i, (block, label) in enumerate(zip(blocks, labels)):

        # M5 → linha contínua, H1 → tracejada
        #if i < 3:
        #    linestyle = '-'
        #    marker = 'o'
        #else:
        #    linestyle = '--'
        #    marker = 's'

        plt.errorbar(
            block['EMBED_DIM'],
            block['ALPHA'],
            yerr=block['ALPHA_STD'],
        #    linestyle=linestyle,
            fmt='o-', ms=5, lw=1.4, elinewidth=0.8, capsize=2,
        #    marker=marker,
            label=label
        )

    # =========================
    # Finalização
    # =========================
    plt.xlabel('EMBED_DIM')
    plt.ylabel('ALPHA')
    plt.title(title)
    plt.legend()
    #plt.grid(True)

    plt.tight_layout()

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def parse_blocks(text):
    blocks = re.split(r'Valor\s+STEP', text)
    results = []

    for block in blocks[1:]:
        data = {}
        #print(block)
        try:
            data['STEP'] = int(re.search(r'(\d+)', block).group(1))
            data['EMBED_DIM'] = int(re.search(r'EMBED_DIM\s+(\d+)', block).group(1))
            data['WINDOW'] = int(re.search(r'WINDOW\s+(\d+)', block).group(1))

            star = re.search(r'\*\s+([\d\.]+)', block)
            data['MEAN_TIME'] = float(star.group(1)) if star else None

            main = re.search(r'\n([\d\.]+)\s±\s([\d\.nan]+)', block)
            if main:
                data['ALPHA'] = float(main.group(1))
                data['ALPHA_STD'] = None if main.group(2) == 'nan' else float(main.group(2))

            clean_block = block.replace('\xa0', ' ')
            lines = [l.strip() for l in clean_block.splitlines() if l.strip()]

            for i, line in enumerate(lines):
                if line in ['RFDT', 'T', 'S', 'B']:
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        nums = re.search(r'([\d\.]+)\s*±\s*([\d\.nan]+)', next_line)
                        if nums:
                            data[line] = float(nums.group(1))
                            data[f'{line}_STD'] = None if nums.group(2) == 'nan' else float(nums.group(2))

            # for key in ['RFDT', 'T', 'S', 'B']:
            #     match = re.search(rf'{key}\s*\n\s*([\d\.]+)\s±\s([\d\.nan]+)', clean_block)
            #     if match:
            #         data[key] = float(match.group(1))
            #         data[f'{key}_STD'] = None if match.group(2) == 'nan' else float(match.group(2))
            #         #print(key, "->", match.groups() if match else None)

            l2 = re.search(r'L2\s+([\d\.]+)', block)
            if l2:
                data['L2'] = float(l2.group(1))

            results.append(data)

        except:
            pass

    return results


def pdf_to_csv(pdf_path, output):
    text = extract_text(pdf_path)
    data = parse_blocks(text)

    df = pd.DataFrame(data)
    #df.sort_values(by="STEP", inplace=True)
    df.to_csv(output, index=False)

    print("CSV gerado com sucesso!")

def collect_data(csv_path):
    df = pd.read_csv(csv_path)
    return df['ALPHA'].iloc[0], df['ALPHA_STD'].iloc[0], df['RFDT'].iloc[0], df['RFDT_STD'].iloc[0], df['T'].iloc[0], df['T_STD'].iloc[0], df['S'].iloc[0], df['S_STD'].iloc[0], df['L2'].iloc[0]

def compute_instability_min(csv_path, mode):
    df = pd.read_csv(csv_path)

    # print("=== CSV COMPLETO ===")
    # print(df.to_string())          # <-- mostra tudo
    # print("=== FIM CSV ===")

    n = len(df) // 2
    df1 = df.iloc[:n].copy()
    df2 = df.iloc[n:].copy()

    df1 = df1.rename(columns={'S': 'S1', 'S_STD': 'S1_STD'})
    df2 = df2.rename(columns={'S': 'S2', 'S_STD': 'S2_STD'})

    merged = pd.merge(df1, df2, on=['STEP', 'EMBED_DIM', 'WINDOW'])

    # print("=== MERGED ===")
    # print(merged.to_string())      # <-- mostra o resultado do merge
    # print("=== FIM MERGED ===")

    # =========================
    # MODO STEP (inalterado)
    # =========================
    # if mode == 'step':
    #
    #     merged = merged.sort_values('STEP').reset_index(drop=True)
    #
    #     instab = []
    #
    #     for i in range(len(merged)):
    #         vals = []
    #
    #         vals += [merged.loc[i, 'S1'], merged.loc[i, 'S2']]
    #
    #         if i > 0:
    #             vals += [merged.loc[i-1, 'S1'], merged.loc[i-1, 'S2']]
    #
    #         if i < len(merged)-1:
    #             vals += [merged.loc[i+1, 'S1'], merged.loc[i+1, 'S2']]
    #
    #         #if len(vals) == 6:
    #         #remove as bordas
    #         if i==0 :instab.append(np.nan)
    #         elif i==(len(merged)-1) :instab.append(np.nan)
    #         else:instab.append(np.var(vals,ddof=1))
    #         #else:
    #         #    instab.append(np.nan)
    #
    #     merged['INSTAB'] = instab
    #
    #     best_row = merged.loc[merged['INSTAB'].idxmin()]
    #     #best_row = instab.loc[instab.idxmin()]
    #
    #     return merged, best_row
    if mode == 'step':

        merged = merged.sort_values('STEP').reset_index(drop=True)
        n = len(merged)

        scores = []

        for i in range(2, n - 2):  # bordas excluídas (2 de cada lado)
            # Instabilidade individual: variância dos 5 pontos centrados em i
            local_S1 = np.var([merged.loc[i-2, 'S1'],
                               merged.loc[i-1, 'S1'],
                               merged.loc[i,   'S1'],
                               merged.loc[i+1, 'S1'],
                               merged.loc[i+2, 'S1']], ddof=1)

            local_S2 = np.var([merged.loc[i-2, 'S2'],
                               merged.loc[i-1, 'S2'],
                               merged.loc[i,   'S2'],
                               merged.loc[i+1, 'S2'],
                               merged.loc[i+2, 'S2']], ddof=1)

            # Termo A: média das instabilidades individuais
            A = (local_S1 + local_S2) / 2.0

            # Termo B: coerência inter-escala
            B = abs(merged.loc[i, 'S1'] - merged.loc[i, 'S2'])

            scores.append({'idx': i, 'A': A, 'B': B})

        scores_df = pd.DataFrame(scores).set_index('idx')

        A_mean = scores_df['A'].mean()
        B_mean = scores_df['B'].mean()

        merged['INSTAB'] = np.nan
        merged.loc[scores_df.index, 'INSTAB'] = (
            scores_df['A'] / A_mean +
            scores_df['B'] / B_mean
        ).values

        best_row = merged.loc[merged['INSTAB'].idxmin()]

        return merged, best_row
        # =========================
    # MODO EMBED (novo)
    # =========================
    elif mode == 'embed':

        embed_values = sorted(merged['EMBED_DIM'].unique())
        n = len(embed_values)

        instab_list = []

        for i, E in enumerate(embed_values):

            # bordas excluídas (precisam de vizinhos dos dois lados)
            if i == 0 or i == n - 1:
                instab_list.append({'EMBED_DIM': E, 'INSTAB': np.nan})
                continue

            E_prev = embed_values[i-1]
            E_next = embed_values[i+1]

            # 3 pontos vizinhos para cada timeframe
            def get_vals(col, dims):
                return merged.loc[merged['EMBED_DIM'].isin(dims), col].values

            s1_local = get_vals('S1', [E_prev, E, E_next])
            s2_local = get_vals('S2', [E_prev, E, E_next])

            # Termo A: instabilidade local de cada curva separadamente
            A = (np.var(s1_local, ddof=1) + np.var(s2_local, ddof=1)) / 2.0

            # Termo B: coerência inter-escala no ponto central
            s1_mid = merged.loc[merged['EMBED_DIM'] == E, 'S1'].values
            s2_mid = merged.loc[merged['EMBED_DIM'] == E, 'S2'].values
            B = abs(s1_mid.mean() - s2_mid.mean())

            instab_list.append({'EMBED_DIM': E, 'INSTAB': None, 'A': A, 'B': B})

        result = pd.DataFrame(instab_list)

        # normalização pela média de cada termo
        A_mean = result['A'].mean(skipna=True)
        B_mean = result['B'].mean(skipna=True)

        result['INSTAB'] = result['A'] / A_mean + result['B'] / B_mean

        best_row = result.loc[result['INSTAB'].idxmin()]

        return result, best_row    # elif mode == 'embed':
    #
    #     embed_values = sorted(merged['EMBED_DIM'].unique())
    #
    #     instab_list = []
    #
    #     for i, E in enumerate(embed_values):
    #
    #         # precisa de vizinhos dos dois lados
    #         if i == 0 or i == len(embed_values) - 1:
    #             instab_list.append({'EMBED_DIM': E, 'INSTAB': np.nan})
    #             continue
    #
    #         E_prev = embed_values[i-1]
    #         E_next = embed_values[i+1]
    #
    #         # subset = merged[
    #         #     merged['EMBED_DIM'].isin([E_prev, E, E_next])
    #         # ]
    #         subset = merged[
    #             merged['EMBED_DIM'].isin([ E])
    #         ]
    #
    #         vals = np.concatenate([
    #             subset['S1'].values,
    #             subset['S2'].values
    #         ])
    #
    #         if len(vals) > 0:
    #             instab = np.var(vals,ddof=1)
    #         else:
    #             instab = np.nan
    #
    #         instab_list.append({'EMBED_DIM': E, 'INSTAB': instab})
    #
    #     result = pd.DataFrame(instab_list)
    #
    #     best_row = result.loc[result['INSTAB'].idxmin()]
    #
    #     return result, best_row

#pdf_to_csv("BGI$D_step.pdf", "saida.csv")# =========================
# EXECUÇÃO
# =========================
""# ATIVOS = ["PETR4","ITUB4","VALE3","WIN$D","ABEV3","WDO$N"]
# # ATIVOS={"T10$D","BGI$D","ITUB4","BIT$D","AAPL34","SEQR11"}
# # ATIVOS={"PETR4"}
# resultados = []
# for asset in ATIVOS:
#     ATIVO=asset
#     print("Calculando para ",ATIVO)
#     # pdf_path = ATIVO+"_step.pdf"
#     # output_csv = ATIVO+"_step.csv"
#     # pdf_to_csv(pdf_path, output_csv)
#     #
#     # #merged, best = compute_dp_and_find_min(output_csv,'step')
#     # merged_step, best_step = compute_instability_min(output_csv,'step')
#     # print("Melhor combinação STEP:")
#     # print(best_step[['STEP', 'EMBED_DIM', 'WINDOW', 'INSTAB']])
#
#     pdf_path = ATIVO+"_embed.pdf"
#     output_csv = ATIVO+"_embed.csv"
#     pdf_to_csv(pdf_path, output_csv)
#
#     merged_embed, best_embed = compute_instability_min(output_csv,'embed')
#     print("Melhor combinação EMBED_DIM:")
#     print(best_embed[['EMBED_DIM', 'INSTAB']])
#
#     pdf_path = ATIVO+"_H1.pdf"
#     output_csv = ATIVO+"_H1+"_results_".csv"
#     pdf_to_csv(pdf_path, output_csv)
#
#     alpha_H1,alpha_std_H1,ratio_fdt_H1,ratio_fdt_std_H1,beta_total_H1,beta_total_std_H1,beta_struct_H1,beta_struct_std_H1,L2_H1 = collect_data(output_csv)
#
#     pdf_path = ATIVO+"_M5.pdf"
#     output_csv = ATIVO+"_M5"+"_results_".csv"
#     pdf_to_csv(pdf_path, output_csv)
#
#     alpha_M5,alpha_std_M5,ratio_fdt_M5,ratio_fdt_std_M5,beta_total_M5,beta_total_std_M5,beta_struct_M5,beta_struct_std_M5,L2_M5 = collect_data(output_csv)
#     #merged_embed, best_embed = compute_instability_min(output_csv,'embed')
#     #print("Melhor combinação EMBED_DIM:")
#     #print(best_embed[['EMBED_DIM', 'INSTAB']])
#
#     resultados.append({
#             'ATIVO': ATIVO,
#             'BEST_STEP': 0,#best_step['STEP'],
#             'BEST_EMBED_DIM': best_embed['EMBED_DIM'],
#             'ALPHA_H1': f"{alpha_H1:.3f} ± {alpha_std_H1:.3f}",
#             'RATIO_H1': f"{ratio_fdt_H1:.3f} ± {ratio_fdt_std_H1:.3f}",
#             'BETA_T_H1': f"{beta_total_H1:.3f} ± {beta_total_std_H1:.3f}",
#             'BETA_S_H1': f"{beta_struct_H1:.3f} ± {beta_struct_std_H1:.3f}",
#             'L2_H1': f"{L2_H1:.3f}",
#             'ALPHA_M5': f"{alpha_M5:.3f} ± {alpha_std_M5:.3f}",
#             'RATIO_M5': f"{ratio_fdt_M5:.3f} ± {ratio_fdt_std_M5:.3f}",
#             'BETA_T_M5': f"{beta_total_M5:.3f} ± {beta_total_std_M5:.3f}",
#             'BETA_S_M5': f"{beta_struct_M5:.3f} ± {beta_struct_std_M5:.3f}",
#             'L2_M5': f"{L2_M5:.3f}"
#         })
#
#
#     #plot_alpha_vs_embed(output_csv,ATIVO)
#
# df_resultados = pd.DataFrame(resultados)
# df_resultados.to_csv("final_results.csv", index=False)
# print("\nTabela final:")
# print(df_resultados)
# #plt.show()
""