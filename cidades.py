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



def gerar_relatorio(df):
    print("\n--- RELATÓRIO AUTOMÁTICO ---")
    if len(df) == 0:
        print("Nenhum dado disponível.")
        return

    ultima = df.iloc[-1]
    print(f"Geração atual: {ultima['generation']}")
    print(f"Número de veículos: {ultima['N_VEHICLES']}")
    print(f"Melhor fitness global: {round(ultima['best_fitness'], 2)}")
    if ultima['N_VEHICLES'] > 1:
        print(f"Fitness individual de cada veículo: {ultima['fitness_veiculos']}")
    print(f"Número de cidades: {len(ultima['cities_locations'])}")
    print("Cidades e coordenadas:")
    for idx, (coord, nome) in enumerate(zip(ultima['cities_locations'], ultima['city_names'])):
        print(f"  {idx+1}: {nome} - {coord}")
    print("Rotas dos veículos:")
    if ultima['N_VEHICLES'] == 1:
        print(f"  Veículo único: {ultima['best_solution']}")
    else:
        for idx, route in enumerate(ultima['best_solution']):
            print(f"  Veículo {idx+1}: {route}")
    print("--- Fim do relatório ---\n")

    # Exemplo de relatório de eficiência
    print(f"Média de fitness das últimas 10 gerações: {df['best_fitness'].tail(10).mean():.2f}")
    print(f"Total de execuções registradas: {len(df)}")
    print("--- Fim do resumo estatístico ---\n")

    # Exemplo: gerar explicação para motoristas usando IA (opcional)
    try:
        import openai
        from openai_key import key
        prompt = (
            f"Explique para os motoristas como será feita a entrega considerando:\n"
            f"- Número de veículos: {ultima['N_VEHICLES']}\n"
            f"- Melhor rota: {ultima['best_solution']}\n"
            f"- Cidades: {', '.join(ultima['city_names'])}\n"
            f"- Fitness individual: {ultima['fitness_veiculos']}\n"
            f"Seja claro, objetivo e amigável."
        )
        openai.api_key = key
        response = openai.chat.completions.create(
            model="gpt-4o", #model="gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        explicacao = response.choices[0].message.content
        print("\n--- Orientação para os motoristas ---")
        print(explicacao)
        print("--- Fim da orientação ---\n")
    except Exception as e:
        print("Erro ao gerar explicação com IA:", e)

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



