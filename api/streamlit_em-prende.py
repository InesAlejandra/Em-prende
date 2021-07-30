import streamlit as st
from PIL import Image
import pandas as pd

img = Image.open(r"api/logo.png")
entradas = []

c = st.beta_columns([2,2])
#c[0].title("Em-prende!!")
c[0].image(img,width = 250)

st.header("¿Quieres averiguar como les va a otros negocios en tu mercado?")

show = st.beta_container()
locations = pd.DataFrame([[-0.2201641, -78.5123274]],columns = ["lat", "lon"]) 
show.map(locations)

# Tamaño negocio
st.header("¿Qué tamaño es tu negocio?")
t = ["","Microempresa","Pequeña empresa","Mediana empresa","Grande empresa"]
t1 = st.selectbox("El tamaño de mi negocio es:",t)
st.info(t1)
if t1 == "Microempresa":
    tmn = 1
elif t1 == "Pequeña empresa":
    tmn = 2
elif t1 == "Mediana empresa":
    tmn = 3
elif t1 == "Grande empresa":
    tmn = 4

st.header("¿Cuál es tu principal área de negocio?")

# Area de negocio
negocios = ["","Agricultura, ganadería, silvicultura y pesca",
            "Explotación de minas y canteras",
            "Industrias manufactureras",
            "Suministro de electricidad, gas, vapor y aire acondicionado",
            "Suministro de agua, evacuación de aguas residuales, gestión de desechos y descontaminación",
            "Construcción",
            "Comercio al por mayor y al por menor; reparación de vehículos automotores y motocicletas",
            "Transporte y almacenamiento",
            "Actividades de alojamiento y de servicio de comidas",
            "Información y comunicaciones",
            "Actividades financieras y de seguros",
            "Actividades inmobiliarias",
            "Actividades profesionales, científicas y técnicas",
            "Actividades de servicios administrativos y de apoyo",
            "Administración pública y defensa; planes de seguridad social de afiliación obligatoria",
            "Enseñanza",
            "Actividades de atención de la salud humana y de asistencia socia",
            "Actividades artísticas, de entretenimiento y recreativas",
            "Otras actividades de servicios",
            "Actividades de los hogares como empleadores",
            "Actividades de organizaciones y órganos extraterritoriales"]

# Area de negocio industria manufacturera
neg_C = ("","Elaboración de productos alimenticios",
        "Elaboración de bebidas",
        "Elaboración de productos de tabaco",
        "Fabricación de productos textiles",
        "Fabricación de prendas de vestir",
        "Fabricación de productos de cuero y productos conexos",
        "Producción de madera y fabricación de productos de madera y corcho excepto muebles",
        "Fabricación de artículos de paja y de materiales trenzables",
        "Fabricación de papel y de productos de papel",
        "Impresión y reproducción de grabaciones",
        "Fabricación de coque y productos de la refinación del petróleo",
        "Fabricación de sustancias y productos químicos",
        "Fabricación de productos farmacéuticos, sustancias químicas medicinales y productos botánicos de uso farmacéutico",
        "Fabricación de productos de caucho y de plástico",
        "Fabricación de otros productos minerales no metálicos",
        "Fabricación de metales comunes",
        "Fabricación de productos elaborados de metal, excepto maquinaria y equipo")

negocio = st.selectbox("Mi área principal de negocio es:",negocios)
st.info(negocio)

if negocio == negocios[3]:
    opt = list(range(len(neg_C)))
    #C = st.selectbox("Industrias manufactureras",neg_C)
    pos_C = st.selectbox("Industrias manufactureras",opt,format_func = lambda x:neg_C[x])
    st.info(neg_C[pos_C])
    
# Prediccion
if st.button("Emprende"):
    st.balloons()
        

