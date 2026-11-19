import math


class Ansur3DTo2DConverter:
    """
    ANSUR II (Anthropometric Survey of US Army Personnel) tabanlı çift cinsiyetli (Kadın & Erkek)
    3D -> 2D antropometrik dönüşüm modeli.
    3 Boyutlu Çevre (Circumference) ölçülerini, 2 Boyutlu Ön Genişliğe (Front Width) ve
    Yan Derinliğe (Side Depth) dönüştürür.
    """

    def __init__(self, gender="Kadın"):
        self.gender = gender
        # Kadın ve Erkek için ANSUR II Derinlik / Genişlik (Depth-to-Width) elips oranları
        self.gender_ratios = {
            "Kadın": {
                "chest": 0.70,
                "waist": 0.65,
                "hips": 0.72,
                "thigh": 0.82,
                "neck": 0.90,
            },
            "Erkek": {
                "chest": 0.64,  # Erkek göğüs kafesi önden daha geniş, derinlik/genişlik oranı daha düşüktür
                "waist": 0.68,
                "hips": 0.69,
                "thigh": 0.85,
                "neck": 0.92,
            },
        }
        self.ratios = self.gender_ratios.get(gender, self.gender_ratios["Kadın"])

    def set_gender(self, gender="Kadın"):
        self.gender = "Erkek" if "erkek" in str(gender).lower() else "Kadın"
        self.ratios = self.gender_ratios[self.gender]

    def circumference_to_width(self, circumference_cm, body_part="chest", gender=None):
        """
        Elips çevre formülü (Ramanujan / Euler yaklaşımı) baz alınarak 3D çevreyi 2D ön genişliğe çevirir.
        Formül: C ≈ pi * sqrt(2 * ((W/2)^2 + (D/2)^2))
        """
        if gender is not None:
            self.set_gender(gender)
        k = self.ratios.get(body_part, 0.70)
        denominator = math.pi * math.sqrt((1.0 + k ** 2) / 2.0)
        width_2d = float(circumference_cm) / denominator
        return width_2d

    def circumference_to_depth(self, circumference_cm, body_part="chest", gender=None):
        """
        Yan profil (Side View) için 3D çevreyi 2D yan derinliğe (D = k * W) dönüştürür.
        """
        w_2d = self.circumference_to_width(circumference_cm, body_part=body_part, gender=gender)
        k = self.ratios.get(body_part, 0.70)
        return w_2d * k
