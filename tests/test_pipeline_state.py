"""
Тесты, демонстрирующие проблемы с глобальным состоянием в классе Pipeline.

Проблемы:
1. self.results хранится как атрибут экземпляра
2. meta_results возвращается по ссылке, а не копией
3. Состояние может быть модифицировано извне, что нарушает инкапсуляцию
"""
import pytest
from pathlib import Path


class TestPipelineStateIssues:
    """Тесты, демонстрирующие проблемы с состоянием Pipeline"""

    @pytest.fixture
    def pipeline(self):
        """Создаёт один экземпляр Pipeline для всех тестов"""
        from russian_docs_ocr.document_processing import Pipeline
        return Pipeline(model_format='ONNX', device='cpu', verbose=False)

    @pytest.fixture
    def sample_images(self):
        """Возвращает пути к тестовым изображениям"""
        samples_dir = Path('samples')
        images = {
            'dl_2011': list(samples_dir.joinpath('DL_2011').glob('*.jpg'))[:1],
        }
        return images

    def test_results_attribute_exposed(self, pipeline, sample_images):
        """
        Демонстрирует проблему: self.results доступен для модификации извне.
        
        Любой код может изменить self.results между вызовами,
        что приведёт к неопределённому поведению.
        """
        dl_images = sample_images.get('dl_2011', [])
        if not dl_images:
            pytest.skip("No DL images")
        
        # Вызываем pipeline
        result = pipeline(dl_images[0], ocr=False, check_quality=False)
        
        # Проблема: self.results доступен напрямую
        # Внешний код может его модифицировать
        original_meta = pipeline.results.meta_results.copy()
        
        # изменяем поле:
        pipeline.results.meta_results['Quality']['Glare'] = 'tampered'
        
        # Теперь quality возвращает изменённые данные
        assert pipeline.results.quality.get('Glare') == 'tampered'
        
        print(f"\nOriginal meta keys: {list(original_meta.keys())}")
        print(f"Current quality: {pipeline.results.quality}")
        print("This violates encapsulation - state can be modified externally!")
        
        # ЭТОТ ТЕСТ ДОЛЖЕН ПАДАТЬ - демонстрируем проблему
        # Если состояние можно модифицировать извне - это баг
        assert 'tampered' not in pipeline.results.quality.get('Glare', ''), \
            "BUG: Internal state was modified from outside - encapsulation violated!"

    def test_meta_results_shared_reference(self, pipeline, sample_images):
        """
        Демонстрирует проблему: meta_results возвращается по ссылке, а не копией.
        
        Пользователь может получить ссылку на внутренний dict и модифицировать его.
        """
        dl_images = sample_images.get('dl_2011', [])
        if not dl_images:
            pytest.skip("No DL images available")
        
        # Вызываем pipeline
        result = pipeline(dl_images[0], ocr=False, check_quality=False)
        
        # Пользователь получает ссылку на meta_results
        meta_ref = result.meta_results
        
        # Сохраняем оригинальное содержимое Quality
        original_quality_content = dict(meta_ref.get('Quality', {}))
        
        # Пользователь может случайно или намеренно модифицировать состояние
        meta_ref['Quality']['UserModified'] = True
        
        # Проверяем, что модификация видна через result
        print(f"\n=== Shared Reference Test ===")
        print(f"Original quality: {original_quality_content}")
        print(f"Modified quality: {result.meta_results['Quality']}")
        
        # ЭТОТ ASSERT ДОЛЖЕН ПАДАТЬ - демонстрируем проблему
        # meta_results возвращается по ссылке, а не копией
        assert result.meta_results['Quality'].get('UserModified') != True, \
            "BUG: meta_results returned by reference, allowing external modification!"
        
        # Дополнительная проверка: проверяем, что copy() не помогает для вложенных dict
        result2 = pipeline(dl_images[0], ocr=False, check_quality=False)
        meta_copy = result2.meta_results.copy()  # поверхностная копия
        meta_copy['Quality']['FromCopy'] = True
        
        # Из-за поверхностной копии, вложенный dict всё ещё общий
        assert 'FromCopy' in result2.meta_results.get('Quality', {}), \
            "BUG: Shallow copy doesn't protect nested dicts - shared state!"

    def test_external_state_modification(self, pipeline, sample_images):
        """
        Демонстрирует проблему: внешний код может модифицировать внутреннее состояние.
        
        Это нарушает инкапсуляцию и может привести к трудноотлавливаемым багам.
        """
        dl_images = sample_images.get('dl_2011', [])
        if not dl_images:
            pytest.skip("No DL images")
        
        # Вызываем pipeline
        result = pipeline(dl_images[0], ocr=False, check_quality=False)
        
        # Получаем доступ к внутреннему состоянию через result
        # Это возможно, потому что meta_results возвращается по ссылке
        internal_meta = result.meta_results
        
        # Сохраняем оригинальное состояние для сравнения
        original_quality_keys = set(result.meta_results.get('Quality', {}).keys())
        
        # Модифицируем состояние извне
        internal_meta['Quality']['Tampered'] = True
        internal_meta['Hacked'] = 'External code modified this!'
        
        # Проверяем, что изменения видны через pipeline
        print(f"\n=== External Modification Test ===")
        print(f"Quality после модификации: {pipeline.results.meta_results['Quality']}")
        print(f"'Hacked' key exists: {'Hacked' in pipeline.results.meta_results}")
        
        # ЭТОТ ТЕСТ ДОЛЖЕН ПАДАТЬ - демонстрируем проблему
        # Если внешний код может модифицировать состояние - это баг
        assert 'Hacked' not in pipeline.results.meta_results, \
            "BUG: External code modified internal state via meta_results reference!"
        
        assert set(result.meta_results.get('Quality', {}).keys()) == original_quality_keys, \
            "BUG: Quality keys were modified from outside!"
        
        print("PROBLEM: External code can modify internal state!")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
