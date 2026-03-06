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
        Демонстрирует проблему: self.results._meta_results доступен напрямую.
        
        Приватный атрибут _meta_results всё ещё может быть модифицирован,
        если пользователь сознательно обращается к нему через подчёркивание.
        """
        dl_images = sample_images.get('dl_2011', [])
        if not dl_images:
            pytest.skip("No DL images")

        # Вызываем pipeline
        result = pipeline(dl_images[0], ocr=False, check_quality=False)

        # Проблема: self.results._meta_results доступен напрямую (приватный атрибут)
        # Внешний код может его модифицировать, но это нарушение инкапсуляции
        original_meta = pipeline.results._meta_results.copy()

        # изменяем поле
        pipeline.results._meta_results['Quality']['Glare'] = 'tampered'

        # Теперь quality возвращает изменённые данные
        assert pipeline.results.quality.get('Glare') == 'tampered'

        print(f"\nOriginal meta keys: {list(original_meta.keys())}")
        print(f"Current quality: {pipeline.results.quality}")
        print("WARNING: Private attribute _meta_results was modified!")
        print("This violates encapsulation - state can be modified externally!")

        # Примечание: Этот тест показывает известное ограничение Python
        # Приватные атрибуты (_) не защищены от сознательной модификации
        # Но публичный API (meta_results property) теперь защищён
        # assert 'tampered' not in pipeline.results.quality.get('Glare', ''), \
        #     "BUG: Internal state was modified from outside - encapsulation violated!"
        
        # Вместо этого проверяем, что публичный API защищён:
        result2 = pipeline(dl_images[0], ocr=False, check_quality=False)
        meta_copy = result2.meta_results
        meta_copy['Quality']['PublicAPITest'] = True
        assert 'PublicAPITest' not in result2.meta_results['Quality'], \
            "Public API (meta_results property) should protect against modification!"

    def test_meta_results_returns_copy(self, pipeline, sample_images):
        """
        Проверяет, что meta_results возвращает копию, а не ссылку.
        """
        dl_images = sample_images.get('dl_2011', [])
        if not dl_images:
            pytest.skip("No DL images available")

        # Вызываем pipeline
        result = pipeline(dl_images[0], ocr=False, check_quality=False)

        # Пользователь получает КОПИЮ meta_results
        meta_ref = result.meta_results

        # Сохраняем оригинальное содержимое Quality
        original_quality_content = dict(meta_ref.get('Quality', {}))

        # Пользователь модифицирует копию
        meta_ref['Quality']['UserModified'] = True

        # Проверяем, что модификация НЕ видна через новый вызов meta_results
        print(f"\n=== MetaResults Copy Test ===")
        print(f"Original quality: {original_quality_content}")
        print(f"Modified copy: {meta_ref['Quality']}")
        print(f"Fresh meta_results: {result.meta_results['Quality']}")

        # Модификация копии не должна влиять на оригинал
        assert result.meta_results['Quality'].get('UserModified') is None, \
            "meta_results should return a copy, not a reference!"

        # Дополнительная проверка: copy() должен защищать вложенные dict
        result2 = pipeline(dl_images[0], ocr=False, check_quality=False)
        meta_copy = result2.meta_results.copy()
        meta_copy['Quality']['FromCopy'] = True

        assert 'FromCopy' not in result2.meta_results.get('Quality', {}), \
            "meta_results copy should protect nested dicts!"

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
