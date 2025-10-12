import pandas as pd
from io import StringIO
import openai
from openai_key import key

def chat_sobre_rotas(df):
    print("\nDigite sua pergunta sobre as rotas e entregas (ou 'sair' para voltar):")
    while True:
        pergunta = input("Pergunta: ")
        if pergunta.lower() in ["sair", "exit", "quit"]:
            break
        # Gera um contexto textual com as últimas 10 gerações
        contexto = df.tail(10).to_string()
        prompt = (
            f"Você é um assistente de logística. Use os dados abaixo para responder à pergunta do usuário.\n"
            f"Dados das últimas gerações:\n{contexto}\n"
            f"Pergunta: {pergunta}\n"
            f"Responda de forma clara, objetiva e baseada nos dados apresentados."
        )
        openai.api_key = key
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400
            )
            resposta = response.choices[0].message.content
            print("\nResposta do sistema:", resposta, "\n")
        except Exception as e:
            print("Erro ao consultar IA:", e)




# ...existing code...
import json
import math

# Quanto 1 unidade do seu plano equivale em quilômetros (ajuste se necessário)
SCALE_KM_PER_UNIT = 1.0

try:
    import openai
    from openai_key import key
except Exception:
    openai = None
# ...existing code...

def gerar_relatorio(df, close_route=True):
    """
    Gera relatório claro e fiel:
    - Veículo X começa em <cidade> e termina em <cidade>
    - Percurso: CidadeA -> CidadeB -> ...
    - Trechos com quilometragem por perna e total por veículo
    - close_route=True fecha o ciclo (volta ao início)
    Ao final, gera um informativo via LLM com dados estruturados e baixa temperatura.
    """
    print("\n--- RELATÓRIO AUTOMÁTICO ---")
    if len(df) == 0:
        print("Nenhum dado disponível.")
        return

    ultima = df.iloc[-1]
    nveh = int(ultima["N_VEHICLES"])
    city_names = list(ultima["city_names"])
    cities_locations = [tuple(c) for c in ultima["cities_locations"]]
    best_solution = ultima["best_solution"]

    # Utilitários locais
    def to_index_route(route):
        """Aceita rota como lista de índices ou de coordenadas; retorna sempre índices."""
        if not route:
            return []
        if isinstance(route[0], int):
            return list(route)
        # rota em coords -> índices (robusto a int/float)
        m = {}
        for i, c in enumerate(cities_locations):
            m[tuple(c)] = i
            m[(int(c[0]), int(c[1]))] = i
        idxs = []
        for p in route:
            t = tuple(p)
            i = m.get(t) or m.get((int(p[0]), int(p[1])))
            if i is None:
                # fallback: índice mais próximo (se coords não baterem 100%)
                px, py = float(p[0]), float(p[1])
                best_i, best_d = None, float("inf")
                for j, (x, y) in enumerate(cities_locations):
                    d = (x - px) ** 2 + (y - py) ** 2
                    if d < best_d:
                        best_i, best_d = j, d
                i = best_i
            idxs.append(i)
        return idxs

    def dist_km(a_idx, b_idx):
        ax, ay = cities_locations[a_idx]
        bx, by = cities_locations[b_idx]
        return math.hypot(bx - ax, by - ay) * SCALE_KM_PER_UNIT

    def km_fmt(v): return f"{v:.1f} km"

    # Normaliza best_solution para lista de rotas
    routes = [best_solution] if nveh <= 1 else list(best_solution)

    print(f"Geração: {int(ultima['generation'])}")
    print(f"Veículos: {nveh}")
    print(f"Cidades na malha: {len(cities_locations)}")
    print("\nRotas detalhadas por veículo:")

    total_geral = 0.0
    all_stats = []  # para IA

    for vidx, route in enumerate(routes, start=1):
        route_idx = to_index_route(route)
        if len(route_idx) < 1:
            print(f"\n- Veículo {vidx}: rota vazia.")
            all_stats.append({"vehicle": vidx, "legs": [], "total_km": 0.0})
            continue

        seq = list(route_idx)
        if close_route and len(seq) >= 2:
            seq_pairs = [(seq[i], seq[i+1]) for i in range(len(seq)-1)] + [(seq[-1], seq[0])]
        else:
            seq_pairs = [(seq[i], seq[i+1]) for i in range(len(seq)-1)]

        legs_km = [dist_km(a, b) for (a, b) in seq_pairs]
        total_km = sum(legs_km)
        total_geral += total_km

        path_names = [city_names[i] for i in seq]
        if close_route and len(seq) >= 2:
            path_line = " -> ".join(path_names + [path_names[0]])
            start_name, end_name = path_names[0], path_names[0]
        else:
            path_line = " -> ".join(path_names)
            start_name, end_name = path_names[0], path_names[-1]

        print(f"\n- Veículo {vidx}:")
        print(f"  Começa em {start_name} e termina em {end_name}.")
        print(f"  Percurso: {path_line}")

        if seq_pairs:
            print("  Trechos:")
            for (a, b), d in zip(seq_pairs, legs_km):
                print(f"    {city_names[a]} -> {city_names[b]}: {km_fmt(d)}")

            km_sequence = ", depois ".join(km_fmt(k) for k in legs_km)
            print(f"  Quilometragem por trecho (na ordem): {km_sequence}")

        print(f"  Total da rota: {km_fmt(total_km)}")

        all_stats.append({
            "vehicle": vidx,
            "legs": [{"from": city_names[a], "to": city_names[b], "km": round(d, 1)}
                     for (a, b), d in zip(seq_pairs, legs_km)],
            "total_km": round(total_km, 1),
            "start": start_name,
            "end": end_name,
            "path": path_names + ([path_names[0]] if close_route and len(seq) >= 2 else [])
        })

    print(f"\nTotal geral planejado: {km_fmt(total_geral)}")
    print("--- Fim do relatório ---\n")

    # --- Informativo via LLM (opcional, com dados estruturados e baixa temperatura) ---
    if openai is None:
        print("OpenAI não configurado; informativo via IA não gerado.")
        return

    try:
        openai.api_key = key
        structured = {
            "generation": int(ultima["generation"]),
            "vehicles": nveh,
            "routes": all_stats,
            "overall_km": round(total_geral, 1),
        }

        system_msg = (
            "Você é um assistente de logística. Gere um briefing para motoristas SOMENTE com base nos dados fornecidos. "
            "Não invente cidades ou trechos. Use a ordem exata das pernas e liste os quilômetros de cada trecho e o total."
        )
        user_msg = (
            "muito importante uma mensagem inicial de boas-vindas e agradecimento ao motorista."
            "Mensagem inicial para o motorista positiva, agradecendo seu trabalho e esforço diário."
            "Produza um informativo claro para motoristas, em português, no formato:\n"
            "- Veículo X: começa em <Cidade>, termina em <Cidade>.\n"
            "  Percurso: CidadeA -> CidadeB -> ...\n"
            "  Trechos: CidadeA -> CidadeB: Y km; ...\n"
            "  Quilometragem por trecho (na ordem): 100.0 km, depois 200.0 km, ...\n"
            "  Total da rota: Z km.\n"
            "Finalize com o total geral. Seja objetivo, sem floreios.\n\n"
            "Finalizar com um agradecimento pelo esforço no caso de problemas ligue para um 0800 da empresa crie informações de segurança.\n\n"
            f"DADOS:\n{json.dumps(structured, ensure_ascii=False)}"
        )

        resp = openai.chat.completions.create(
            model="gpt-4o",          # use gpt-3.5-turbo se preferir
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=900,
            temperature=0.2,
        )
        texto = resp.choices[0].message.content

        print("\n--- Informativo para motoristas (IA) ---")
        print(texto)
        print("--- Fim do informativo ---\n")

        # Salva no df (última linha)
        try:
            df.at[df.index[-1], "LLM_summary"] = texto
        except Exception:
            pass

    except Exception as e:
        print("Erro ao gerar informativo com IA:", e)
# ...existing code...
# Dados em formato de texto (copiados da sua mensagem)
dados = """ID	Cidade	População
1	Recife	1587707
2	Jaboatão dos Guararapes	683285
3	Petrolina	414083
4	Caruaru	402290
5	Olinda	365402
6	Paulista	362960
7	Cabo de Santo Agostinho	216969
8	Camaragibe	155771
9	Garanhuns	151064
10	Vitória de Santo Antão	143799
11	Igarassu	122312
12	São Lourenço da Mata	117759
13	Ipojuca	105638
14	Santa Cruz do Capibaribe	104277
15	Abreu e Lima	103945
16	Serra Talhada	98143
17	Gravatá	91887
18	Araripina	90104
19	Goiana	85160
20	Belo Jardim	83647
21	Carpina	83205
22	Arcoverde	82003
23	Ouricuri	68329
24	Surubim	67515
25	Salgueiro	65635
26	Pesqueira	65408
27	Bezerros	64809
28	Escada	62252
29	Paudalho	59638
30	Limoeiro	59125
31	Moreno	57647
32	Palmares	56615
33	Buíque	54425
34	São Bento do Una	51264
35	Brejo da Madre de Deus	51107
36	Timbaúba	47575
37	Bom Conselho	46192
38	Águas Belas	43713
39	Toritama	43636
40	Santa Maria da Boa Vista	42682
41	Afogados da Ingazeira	42407
42	Barreiros	42056
43	Lajedo	41786
44	Custódia	39403
45	Bom Jardim	39278
46	Sirinhaém	39233
47	Bonito	39163
48	São Caitano	39117
49	Aliança	37372
50	São José do Belmonte	36752
51	Itambé	36626
52	Bodocó	36129
53	Petrolândia	35991
54	Sertânia	34269
55	Ribeirão	34255
56	Itaíba	33691
57	Exu	33436
58	Catende	33279
59	São José do Egito	32491
60	Nazaré da Mata	32153
61	Trindade	32086
62	Cabrobó	31746
63	Floresta	31627
64	Ipubi	30603
65	Caetés	30441
66	Glória do Goitá	30370
67	Passira	29719
68	Itapissuma	29463
69	Tabira	29093
70	João Alfredo	28903
71	Ibimirim	28760
72	Inajá	27488
73	Vicência	27297
74	Água Preta	27221
75	Tupanatinga	27009
76	Pombos	26847
77	Manari	26773
78	Ilha de Itamaracá	25529
79	Taquaritinga do Norte	25497
80	Condado	25383
81	Canhotinho	25090
82	Lagoa Grande	24952
83	Tacaratu	24803
84	São João	24772
85	Macaparana	24624
86	Agrestina	24615
87	Tamandaré	24534
88	Cupira	24301
89	Pedra	23605
90	Panelas	23449
91	Vertentes	22955
92	Orobó	22438
93	Feira Nova	22169
94	Riacho das Almas	21411
95	Chã Grande	21224
96	Altinho	21185
97	Flores	20835
98	Cachoeirinha	20612
99	Rio Formoso	20460
100	São Joaquim do Monte	20440
101	Araçoiaba	19936
102	Lagoa de Itaenga	19915
103	Carnaíba	19513
104	São José da Coroa Grande	19468
105	Afrânio	19349
106	Parnamirim	19028
107	Sanharó	18933
108	Capoeiras	18890
109	Serrita	18759
110	Belém do São Francisco	18713
111	Lagoa do Carro	18708
112	Amaraji	18471
113	Camocim de São Félix	17991
114	Quipapá	17974
115	Gameleira	17973
116	Dormentes	17749
117	Correntes	17660
118	Venturosa	17609
119	Iati	17605
120	São Vicente Ferrer	17176
121	Itaquitinga	17109
122	Jataúba	16323
123	Cumaru	16252
124	Jupi	15943
125	Ferreiros	15794
126	Triunfo	15142
127	Mirandiba	14599
128	Santa Maria do Cambucá	14533
129	Jatobá	14463
130	Tracunhaém	14393
131	Lagoa dos Gatos	14386
132	Alagoinha	14335
133	Primavera	14351
134	Santa Cruz	14320
135	Tacaimbó	14277
136	Itapetim	14232
137	Saloá	14173
138	Orocó	14100
139	Frei Miguelinho	14070
140	Jurema	14027
141	Joaquim Nabuco	13503
142	Casinhas	13489
143	São Benedito do Sul	13479
144	Chã de Alegria	13431
145	Buenos Aires	13254
146	Paranatama	12712
147	Carnaubeira da Penha	12682
148	Barra de Guabiraba	12616
149	Santa Filomena	12402
150	Lagoa do Ouro	12364
151	Betânia	11981
152	Jucati	11975
153	Santa Cruz da Baixa Verde	11933
154	Xexéu	11791
155	Machados	11471
156	Calçado	11445
157	Iguaracy	11366
158	Sairé	11218
159	Moreilândia	10849
160	Cedro	10845
161	Belém de Maria	10829
162	Poção	10805
163	Angelim	10580
164	Santa Terezinha	10513
165	Cortês	10512
166	Jaqueira	10483
167	Verdejante	9474
168	Maraial	9432
169	Brejão	9399
170	Terra Nova	9221
171	Tuparetama	8252
172	Brejinho	8010
173	Camutanga	7972
174	Vertente do Lério	7782
175	Ibirajuba	7344
176	Granito	7206
177	Palmeirina	7151
178	Terezinha	6857
179	Quixaba	6755
180	Salgadinho	5620
181	Solidão	5403
182	Calumbi	5367
183	Ingazeira	4959
184	Itacuruba	4490
185	Fernando de Noronha	3316
"""

# Criação do DataFrame
df = pd.read_csv(StringIO(dados), sep='\t')


# Exibir as 10 primeiras linhas
print(df.head(10))

# Verificar tipo das colunas
print(df.dtypes)



