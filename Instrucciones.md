# Instrucciones para Contribuir al Repositorio

## 📋 Resumen del Proceso Realizado

Este documento explica cómo se corrigió y contribuyó al proyecto **Conversor de Temperaturas con IA**. El proceso incluye la corrección de errores, mejora del modelo y subida de cambios al repositorio original.

---

## 🔧 Problemas Corregidos en el Proyecto

### 1. **Dependencias Faltantes**
- **Problema**: `ModuleNotFoundError: No module named 'sklearn'`
- **Solución**: Instalación de bibliotecas necesarias
```bash
python3 -m pip install scikit-learn tensorflow matplotlib numpy
```

### 2. **Arquitectura del Modelo Sobredimensionada**
- **Problema**: Modelo con 3 capas y dropout innecesario
- **Solución**: Simplificación a 2 capas, eliminación de dropout
- **Resultado**: Mejor rendimiento y convergencia más rápida

### 3. **Preprocesamiento Inadecuado**
- **Problema**: Escalado conjunto de temperatura y tipo de conversión
- **Solución**: Escalado separado para temperaturas
- **Resultado**: Mejor precisión del modelo

### 4. **Entrenamiento Insuficiente**
- **Problema**: Solo 50 épocas de entrenamiento
- **Solución**: Incremento a 200 épocas
- **Resultado**: Error reducido de 36.16% a 2.23%

---

## 🚀 Proceso de Contribución (Para Contribuidores)

### Para Usuarios **Linux** (Terminal Git)

#### Paso 1: Configurar Git
```bash
# Configurar tu identidad
git config --global user.name "tu-usuario"
git config --global user.email "tu-email@example.com"

# Verificar configuración
git config --global --list | grep user
```

#### Paso 2: Clonar el Repositorio
```bash
git clone https://github.com/rickgrisalesdev/windsurf-project-3.git
cd windsurf-project-3
```

#### Paso 3: Crear Nueva Rama
```bash
# Crear y cambiar a tu rama de trabajo
git checkout -b nombre-de-tu-rama

# Ejemplo:
git checkout -b mejora-conversor-temperatura
```

#### Paso 4: Realizar Cambios
```bash
# Editar archivos según necesites
# nano app.py o tu editor preferido

# Agregar cambios al staging
git add .

# Hacer commit con mensaje descriptivo
git commit -m "descripción detallada de tus cambios"
```

#### Paso 5: Subir Cambios al Repositorio
```bash
# Para contribuidores directos (con permisos)
git push --set-upstream origin nombre-de-tu-rama

# Si usas SSH (recomendado)
git remote set-url origin git@github.com:rickgrisalesdev/windsurf-project-3.git
git push --set-upstream origin nombre-de-tu-rama
```

#### Paso 6: Crear Pull Request
1. Ve al enlace que GitHub te proporciona (generalmente aparece en la terminal)
2. O visita manualmente: `https://github.com/rickgrisalesdev/windsurf-project-3/pull/new/nombre-de-tu-rama`
3. Revisa tus cambios
4. Añade descripción detallada
5. Crea la Pull Request

---

### Para Usuarios **Windows** (GitHub Desktop)

#### Paso 1: Instalar GitHub Desktop
1. Descarga desde: https://desktop.github.com/
2. Instala y configura tu cuenta GitHub

#### Paso 2: Clonar el Repositorio
1. Abre GitHub Desktop
2. Ve a `File` > `Clone Repository`
3. Busca: `rickgrisalesdev/windsurf-project-3`
4. Elige ubicación local y haz clic en `Clone`

#### Paso 3: Crear Nueva Rama
1. En GitHub Desktop, haz clic en `Current branch`
2. Selecciona `New branch`
3. Nombra tu rama (ej: `mejora-conversor-temperatura`)
4. Haz clic en `Create branch`

#### Paso 4: Realizar Cambios
1. Abre la carpeta del proyecto en tu editor preferido
2. Realiza los cambios necesarios en los archivos
3. Guarda los cambios

#### Paso 5: Hacer Commit
1. Vuelve a GitHub Desktop
2. Verás los cambios listados en la izquierda
3. Escribe un mensaje de commit descriptivo
4. Haz clic en `Commit to nombre-de-tu-rama`

#### Paso 6: Subir Cambios
1. Haz clic en `Push origin` (arriba derecha)
2. Espera a que se suban los cambios

#### Paso 7: Crear Pull Request
1. Después del push, GitHub Desktop mostrará un botón `Create Pull Request`
2. Haz clic en él y se abrirá tu navegador
3. Revisa los cambios y añade descripción
4. Crea la Pull Request

---

## 🔐 Configuración de Autenticación

### Para Linux (SSH - Recomendado)
```bash
# Generar clave SSH
ssh-keygen -t ed25519 -C "tu-email@example.com"

# Iniciar agente SSH
eval "$(ssh-agent -s)"

# Agregar clave SSH
ssh-add ~/.ssh/id_ed25519

# Copiar clave pública
cat ~/.ssh/id_ed25519.pub
```

Luego:
1. Ve a GitHub.com > Settings > SSH and GPG keys
2. Haz clic en `New SSH key`
3. Pega tu clave pública
4. Guarda

### Para Windows (GitHub Desktop)
GitHub Desktop maneja la autenticación automáticamente. Solo necesitas:
1. Iniciar sesión con tu cuenta GitHub
2. El programa gestionará los tokens de acceso

---

## 📝 Buenas Prácticas para Contribuciones

### 1. **Nombres de Ramas**
- Usa nombres descriptivos y en inglés o español
- Ejemplos: `fix-temperature-conversion`, `mejora-modelo-ia`, `add-documentation`

### 2. **Mensajes de Commit**
- Sé claro y conciso
- Usa el tiempo presente: "fix" en lugar de "fixed"
- Ejemplo: `Fix temperature conversion accuracy` o `Corregir precisión de conversión`

### 3. **Pull Requests**
- Añade descripción detallada
- Explica el problema y la solución
- Mencionar pruebas realizadas
- @mencionar a revisores si es necesario

### 4. **Antes de Contribuir**
```bash
# Actualizar tu repositorio con cambios recientes
git pull origin main

# Resolver conflictos si existen
git status
```

---

## 🐛 Solución de Problemas Comunes

### Error: "Permission denied"
```bash
# Verificar usuario configurado
git config user.name
git config user.email

# Si es incorrecto, corregir
git config --global user.name "tu-usuario-correcto"
git config --global user.email "tu-email-correcto@example.com"
```

### Error: "Repository not found"
```bash
# Verificar remote correcto
git remote -v

# Si es incorrecto, corregir
git remote set-url origin https://github.com/rickgrisalesdev/windsurf-project-3.git
```

### Conflictos al hacer Pull
```bash
# Stash tus cambios temporalmente
git stash

# Pull actual
git pull origin main

# Aplicar tus cambios
git stash pop

# Resolver conflictos manualmente
# Luego hacer commit y push
```

---

## 📊 Resumen de Mejoras del Proyecto

### Antes de las Correcciones:
- **Error relativo**: 36.16%
- **Error absoluto**: 7.79 grados
- **Arquitectura**: 3 capas con dropout
- **Entrenamiento**: 50 épocas

### Después de las Correcciones:
- **Error relativo**: 2.23%
- **Error absoluto**: 0.42 grados
- **Arquitectura**: 2 capas optimizadas
- **Entrenamiento**: 200 épocas

### Archivos Modificados:
- `app.py`: Mejoras en modelo y preprocesamiento
- `informe.md`: Documentación completa del proyecto
- `Instrucciones.md`: Este documento

---

## 🎯 Próximos Pasos

1. **Revisar la Pull Request**: Esperar aprobación de los mantenedores
2. **Responder Feedback**: Realizar cambios solicitados si es necesario
3. **Fusión**: Una vez aprobada, los cambios se integrarán al main
4. **Limpiar**: Opcionalmente eliminar tu rama local después de la fusión

---

## 📞 Soporte

Si tienes problemas durante el proceso:

1. **Verifica tu conexión a internet**
2. **Confirma tus permisos en el repositorio**
3. **Revisa la sintaxis de los comandos**
4. **Consulta la documentación de GitHub**: https://docs.github.com/

---

**¡Feliz contribución! 🚀**
