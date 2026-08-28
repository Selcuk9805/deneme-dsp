# Automix DSP Backend

Automix DSP Backend, müzik çalarlar (özellikle `flutter_soloud` veya benzeri oyun/ses motorları) için iki ses dosyası arasında **profesyonel, at nalı (phasing) etkisi yaratmayan, sample-hassasiyetinde beat eşleştirmeli geçişler (crossfade/automix)** planlamak amacıyla geliştirilmiş bir Python API'sidir.

Bu backend asla sesi *işleyip (render edip)* geri döndürmez. Sadece ses dosyalarını (veya YouTube linklerini) inceler, spektral analiz yapar (vuruşlar, BPM, RMS enerji, low-frequency maskeleme vb.) ve oynatıcı (player) tarafında doğrudan çalıştırılabilecek (executable) bir **Geçiş Planı (Transition Plan)** döndürür.

## Özellikler

- **Dinamik Vuruş Eşleştirme (Beat Matching):** İki şarkının BPM'leri birbirine yakınsa, hedef BPM'i bulur ve her iki parça için hız (speed) oranları (ratio) üretir.
- **Müzikal Cümle (Phrase/Bar) Hizalaması:** Geçiş sürelerini rastgele zamanlara değil, 4 vuruşluk müzikal barlara ve cümlelere (phrase) oturtarak dinleyicinin ritim algısını kırmaz.
- **LUFS Algısal Ses Dengelemesi (Loudness Matching):** EBU R128 standardına göre iki şarkının gerçek algısal gürültüsünü hesaplar ve Flutter tarafında volümleri eşitlemek için `lufs_gain_db` ofsetleri üretir.
- **Camelot Wheel Harmonik Uyum:** Şarkıların notalarını (örneğin C Minor / 5A) algılar. Eğer iki parça müzikal olarak uyumluysa (aynı nota veya komşu) uzun bir harmonik geçiş yapar; uyumsuzsa çamurlaşmayı önlemek için hızlı bir geçiş (dissonant fast cut) stratejisi belirler.
- **Dinamik EQ (Biquad Filter) Otomasyonu:** "At nalı" etkisini (phasing) ve bas frekanslarındaki çamurlaşmayı (muddy bass) engellemek için, iki parçanın bass enerjisini analiz eder. Çakışma durumunda dinamik bir High-Pass filtre frekansı hesaplar ve fade otomasyonu önerir.
- **Zaman Bükülmesi (Time-Warping) Çözünürlüğü:** BPM eşitlemesi amacıyla değişen oynatma hızlarının (speed ratio) otomasyon sürelerine olan etkisini hesaplayarak mutlak `execution_time` değerleri döner. Oynatıcı tarafında sürüklenme (drift) yaşanmaz.
- **Akıllı Önbellekleme (Caching & SQLite):** Ağır ses analizlerini her seferinde tekrarlamamak için indirilen dosyaları MD5 ile hash'leyip önbelleğe alır. Üretilen geçiş planlarını yerleşik SQLite veritabanına kaydeder; aynı iki şarkı istendiğinde milisaniyeler içinde eski JSON planını döner.

---

## Kurulum ve Çalıştırma

**Gereksinimler:**
- Python 3.10+ (Önerilen)
- `ffmpeg` (yt-dlp'nin YouTube videolarından sesi wav olarak ayıklayabilmesi için sisteminizde kurulu olmalıdır)

```bash
# 1. Sanal ortam (venv) oluşturun ve aktif edin
python3 -m venv venv
source venv/bin/activate

# 2. Gereksinimleri yükleyin
pip install -r requirements.txt

# 3. Sunucuyu başlatın (Host ve Port değerlerini kendi ağınıza göre değiştirebilirsiniz)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Sunucu başladığında API dokümantasyonunu incelemek ve test etmek için `http://127.0.0.1:8000/docs` adresini ziyaret edebilirsiniz.

---

## API Entegrasyon Rehberi

Oynatıcınız (örneğin Flutter uygulaması), çalan şarkı (Track A) bitmeye yaklaştığında (örneğin bitimine 30 saniye kala) arka planda bu API'yi çağırarak bir geçiş planı talep etmelidir.

### Geçiş Planı İsteme (POST)

**Endpoint:** `POST /api/transition/plan`

```json
{
  "track_a": "https://www.youtube.com/watch?v=...", // Şu an çalan parça (URL veya Lokal dosya yolu)
  "track_b": "/storage/emulated/0/Music/siradaki_sarki.mp3" // Sıradaki parça
}
```

### Önbelleği (Cache) Temizleme (POST)

**Endpoint:** `POST /api/cache/clear`

Zamanla diskte birikecek olan hash'lenmiş ses dosyalarını (`cache/audio/`) ve SQLite veritabanı analiz geçmişini (`cache/plans.db`) sıfırlar. 

```bash
curl -X POST http://localhost:8000/api/cache/clear
```

### Yanıt (Response) ve Zamanlama Koordinatları

API yanıtında 3 farklı zaman ekseni (timeline) kullanılır. Oyuncu yazılımını entegre ederken bu tanımlara dikkat edilmesi şarttır:

1. `source_time`: Müzik dosyasının *orijinal, hiç hızlandırılmamış* iç zamanı (saniye). Dosyanın neresinden başlanacağını (`seek`) belirtmek için kullanılır.
2. `execution_time`: Player tarafındaki **çalışma (playback) zamanı**. `setRelativePlaySpeed` uygulandıktan sonraki, player'ın scheduling motorunun (örn. SoLoud'un `fadeVolume` süresi) anlayacağı mutlak süredir.

**Örnek Çıktı:**

`confidence` ve `alignment_confidence`, sabit değerler değil — sırasıyla en iyi adayın bir
sonrakine olan skor farkından ve downbeat (bar) tespitinin ne kadar net olduğundan hesaplanır;
gerçek isteklerde tipik olarak ~0.5-0.7 aralığında dağılır, aşağıdaki örnek tek bir isteğin
sonucudur.

```json
{
  "status": "success",
  "schema_version": 3,
  "decision": {
    "strategy": "standard_crossfade",
    "selected_candidate_id": "A104_B0",
    "score": 0.72,
    "confidence": 0.58,
    "scores": {
      "tempo_compatibility": 0.264,
      "key_compatibility": 0.8,
      "phrase_compatibility": 1.0,
      "energy_compatibility": 0.901,
      "bass_compatibility": 0.3
    },
    "candidates": [ ... ]
  },
  "sync": {
    "target_bpm": 126.73,
    "track_a_speed_ratio": 1.0789,
    "track_b_speed_ratio": 0.9318,
    "beat_alignment": {
      "track_a_beat_sample": 1260202,
      "track_b_beat_sample": 1536,
      "alignment_confidence": 0.53
    }
  },
  "timing": {
    "transition_duration_source": 2.0434,
    "transition_duration_execution": 1.8938,
    "track_a_start_crossfade_source": 57.152,
    "track_b_start_source": 0.0697,
    "track_b_play_delay_execution": 0.0
  },
  "automation": {
    "track_a": {
      "lufs_gain_db": -4.55,
      "camelot_key": "8A",
      "volume": [
        {"execution_time": 0.0, "value": 1.0, "type": "set", "curve": "linear"},
        {"execution_time": 1.8938, "value": 0.0, "type": "fadeVolume", "curve": "equal_power"}
      ],
      "biquad_filters": [
        {"filter_type": "highpass", "parameter": "frequency", "execution_time": 0.0, "value": 70.0, "type": "set", "curve": "linear"},
        {"filter_type": "highpass", "parameter": "frequency", "execution_time": 1.8938, "value": 150.0, "type": "fadeFilterParameter", "curve": "linear"}
      ]
    },
    "track_b": {
      "lufs_gain_db": -6.60,
      "camelot_key": "7A",
      "volume": [
        {"execution_time": 0.0, "value": 0.0, "type": "set", "curve": "linear"},
        {"execution_time": 1.8938, "value": 1.0, "type": "fadeVolume", "curve": "equal_power"}
      ],
      "biquad_filters": []
    }
  }
}
```

### Bir Oynatıcıya (Player) Uygulama Adımları

Yukarıdaki JSON'u alan bir Flutter (veya herhangi bir dil) oynatıcısının yapması gereken işlemler sırasıyla şunlardır:

**1. Hızları ve LUFS (Ses Seviyesi) Dengelemesini Ayarlama**
Şarkıların tempo uyuşmazlığını gidermek için `speed_ratio` değerlerini ayarlayın. Ayrıca müzikal bütünlük için backend'in hesapladığı `lufs_gain_db` değerini matematiksel olarak dönüştürerek SoLoud'a verin.
```dart
// SoLoud Örneği
soLoud.setRelativePlaySpeed(trackA_handle, json['sync']['track_a_speed_ratio']); // 1.0789

// LUFS db offset'ini volume multiplier'a çevirme
double aVolumeScale = pow(10, json['automation']['track_a']['lufs_gain_db'] / 20.0);
soLoud.setVolume(trackA_handle, aVolumeScale);
```

**2. Track B'yi Hazırlama (Pre-load)**
Track B dosyasını belleğe yükleyin ancak henüz başlatmayın.

**3. Geçiş Anını (Crossfade Start) Bekleme**
Track A çalmaya devam ediyor. API'nin döndüğü sample-accurate referans noktası `json['sync']['beat_alignment']['track_a_beat_sample']` değerine (veya bunu saniyeye çevirip okuduğunuz playback süresine) ulaşıldığında **GEÇİŞ (TRANSITION) TETİKLENİR**.

**4. Geçiş Tetiklendiğinde Yapılacaklar:**

**Track B'yi Başlat:**
Backend B'nin hangi sample'dan başlayacağını da hesaplamıştır (`track_b_beat_sample`).

> [!WARNING]
> **SoLoud Otomatik Durdurma Davranışı:** SoLoud motoru, bir sesin volume değeri tam `0.0` yapıldığında optimizasyon amacıyla o sesi otomatik olarak durdurabilir (stop/pause). Bu yüzden Track B'yi başlatırken başlangıç sesini `0.0` yerine insan kulağının duyamayacağı `0.001` gibi çok düşük bir değer yapmak bu sorunu çözecektir.

```dart
// Volume listesinde initial volume 0 olarak belirtilse de SoLoud optimizasyonunu aşmak için 0.001 veriyoruz
soLoud.play(trackB_sound, volume: 0.001, playSpeed: json['sync']['track_b_speed_ratio']);
// B'nin başlayacağı sample'a atlama
soLoud.seek(trackB_handle, json['sync']['beat_alignment']['track_b_beat_sample'] / SAMPLE_RATE); 
```

**Track A & Track B Otomasyonlarını (DSP Executor) Uygulama:**
Oynatıcı (Player) karar mekanizmasını umursamaz, doğrudan API'nin verdiği `automation` (volume, biquad_filters) bloklarını okur. Her `event` içindeki `execution_time` ve `curve` parametrelerine uygun bir tween (örneğin linear veya equal_power envelope) çalıştırır.
```dart
// Track A Volume fade-out
double duration = json['timing']['transition_duration_execution']; // 1.89 sn
soLoud.fadeVolume(trackA_handle, 0.0, Duration(milliseconds: (duration * 1000).toInt()));

// Track A Dynamic EQ
soLoud.setFilterParameter(trackA_handle, filterType, paramFreq, 70.0);
soLoud.fadeFilterParameter(trackA_handle, filterType, paramFreq, 150.0, duration);

// Track B Volume fade-in
soLoud.fadeVolume(trackB_handle, 1.0, Duration(milliseconds: (duration * 1000).toInt()));
```



**5. Geçiş Bittiğinde (Cleanup)**
`transition_duration_execution` süresi dolduğunda (örn. 1.89 saniye) Track A'nın sesi tamamen 0.0 olmuş olacaktır. Track A'yı durdurup (`scheduleStop` veya manuel stop) bellekten silebilirsiniz. Müzik artık Track B üzerinden devam etmektedir.
