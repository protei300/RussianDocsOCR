package net.russiandocs.docproc

import net.russiandocs.docproc.pipeline.FieldChecks
import net.russiandocs.docproc.pipeline.FieldChecks.MISMATCH
import net.russiandocs.docproc.pipeline.FieldChecks.UNCHECKED
import net.russiandocs.docproc.pipeline.FieldChecks.VERIFIED
import net.russiandocs.docproc.pipeline.FieldVerdict
import net.russiandocs.docproc.pipeline.MrzChecks
import net.russiandocs.docproc.pipeline.Results
import java.time.LocalDate
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Per-field verdicts — the cases of `tests/test_field_checks.py` (`validation/fields.py`). Every check has a
 * negative control: a deliberately wrong value must come out as `mismatch`. A check that cannot show red
 * guards nothing.
 *
 * The zones are the fictitious documents of `tests/test_mrz_validation.py`; the page values next to them are what
 * those zones say, so a consistent document is the starting point and each test changes exactly one thing.
 */
class FieldChecksTests {

    private val extMrz = "P<RUSROGOVA<<IRINA<<<<<<<<<<<<<<<<<<<<<<<<<<\n7648201572RUS8803148F3105205<<<<<<<<<<<<<<04"
    private val intMrz = "PNRUSLE4KIN<<PAVEL<PETROVI3<<<<<<<<<<<<<<<<<\n4503182940RUS8704234M<<<<<<<7070515770021<46"

    /** The document whose printed number differs from its valid zone in the last digit: it must come out as a mismatch. */
    private val extOcr = mapOf(
        "MRZ" to extMrz,
        "Licence_number" to "76 4820154",
        "Birth_date" to "14.03.1988",
        "Expiration_date" to "20.05.2031",
        "Last_name_en" to "ROGOVA",
        "First_name_en" to "IRINA",
        "Sex_en" to "F",
    )
    private val extOcrConsistent = extOcr + ("Licence_number" to "76 4820157")

    private val intOcr = mapOf(
        "MRZ" to intMrz,
        "Licence_number" to "45 07 318294",
        "Birth_date" to "23.04.1987",
        "Issue_date" to "15.05.2007",
        "Issue_organisation_code" to "770-021",
        "Sex_ru" to "МУЖ.",
        "Last_name_ru" to "ЛЕШКИН",
        "First_name_ru" to "ПАВЕЛ",
        "Middle_name_ru" to "ПЕТРОВИЧ",
        "Birth_place_ru" to "ГОР. МОСКВА",
    )

    private val today = LocalDate.of(2026, 10, 19)

    private fun checks(ocr: Map<String, String>, docType: String?): Map<String, FieldVerdict> =
        FieldChecks.fieldChecks(ocr, docType, today)

    private fun statuses(ocr: Map<String, String>, docType: String?): Map<String, String> =
        checks(ocr, docType).mapValues { it.value.status }

    private fun idsOf(verdict: FieldVerdict): Set<Pair<String, String>> =
        verdict.checks.map { it.id to it.result }.toSet()

    // --- shape ---------------------------------------------------------------

    @Test
    fun everyFieldGetsAnEntryEvenEmptyOnes() {
        val ocr = intOcr + ("Living_region_ru" to "")
        val result = checks(ocr, "INTPASSPORT_2011")
        assertEquals(ocr.keys, result.keys)
        assertEquals(FieldVerdict(UNCHECKED, emptyList(), emptyList()), result["Living_region_ru"])
    }

    @Test
    fun readingIsNeverChanged() {
        val ocr = LinkedHashMap(intOcr + ("Licence_number" to "45 07 318295"))
        val before = LinkedHashMap(ocr)
        checks(ocr, "INTPASSPORT_2011")
        assertEquals(before, ocr)
    }

    @Test
    fun unmeasuredFamiliesAreUncheckedThroughout() {
        for (docType in listOf("INTPASSPORTADDR_ALL", "DLBACK_ALL", "BIRTHCERT_2018", "STS_2019", "NONE", null)) {
            val result = statuses(mapOf("Birth_date" to "31.02.2020", "MRZ" to intMrz), docType)
            assertEquals(setOf(UNCHECKED), result.values.toSet(), docType)
        }
    }

    @Test
    fun familyDoesNotConfuseTheRegistrationPageOrTheLicenceBack() {
        assertEquals("INTPASSPORT", FieldChecks.family("INTPASSPORT_2011"))
        assertNull(FieldChecks.family("INTPASSPORTADDR_ALL"))
        assertEquals("DL", FieldChecks.family("DL_2020"))
        assertNull(FieldChecks.family("DLBACK_ALL"))
        assertEquals("EXTPASSPORT", FieldChecks.family("EXTPASSPORTBIO_2007"))
    }

    // --- zone helpers (mrz.py) -----------------------------------------------

    @Test
    fun checkDigitMatchesIcaoWeights() {
        assertEquals(2, MrzChecks.checkDigit("764820157"))
        assertEquals(0, MrzChecks.checkDigit("<<<<<<<<<<<<<<"))
        assertEquals(null, MrzChecks.checkDigit("76482015!"))
    }

    @Test
    fun validateAndParseTheRealZones() {
        assertTrue(MrzChecks.validateMrz(extMrz).ok)
        val internal = MrzChecks.validateMrz(intMrz)
        assertTrue(internal.ok)
        assertEquals("not_applicable", internal.checks["expiry_date"])
        assertEquals(false, MrzChecks.validateMrz(extMrz.dropLast(1)).ok)
        assertEquals("expected 2 lines", MrzChecks.validateMrz(extMrz.lines()[0]).error)

        val ext = MrzChecks.parseMrz(extMrz)
        assertEquals("ROGOVA", ext.surname)
        assertEquals("IRINA", ext.givenNames)
        assertEquals("14.03.1988", ext.birthDate)
        assertEquals("20.05.2031", ext.expiryDate)
        val int = MrzChecks.parseMrz(intMrz)
        assertEquals("15.05.2007", int.issueDate)
        assertEquals("770021", int.subdivisionCode)
        assertEquals("4507318294", MrzChecks.printedNumber(int))
    }

    // --- internal passport zone ----------------------------------------------

    @Test
    fun consistentInternalPassportIsVerifiedFieldByField() {
        val result = statuses(intOcr, "INTPASSPORT_2011").toMutableMap()
        assertEquals(UNCHECKED, result.remove("Birth_place_ru")) // nothing to compare it with
        assertEquals(setOf(VERIFIED), result.values.toSet())
    }

    @Test
    fun internalNumberIsRebuiltFromTwoPlacesOfTheZone() {
        val verdict = checks(intOcr, "INTPASSPORT_2011").getValue("Licence_number")
        assertTrue(("MRZ-2" to "ok") in idsOf(verdict))
    }

    @Test
    fun internalNumberNegativeControls() {
        // a digit of the number, and the series' fourth digit (lives in the optional data)
        for (wrong in listOf("45 07 318295", "45 08 318294")) {
            val verdict = checks(intOcr + ("Licence_number" to wrong), "INTPASSPORT_2011").getValue("Licence_number")
            assertEquals(MISMATCH, verdict.status, wrong)
            assertEquals("4507318294", verdict.checks.first { it.id == "MRZ-2" }.reference)
        }
    }

    @Test
    fun internalIssueDateAndCodeNegativeControls() {
        val result = statuses(
            intOcr + ("Issue_date" to "16.05.2007") + ("Issue_organisation_code" to "770-022"),
            "INTPASSPORT_2011",
        )
        assertEquals(MISMATCH, result["Issue_date"])
        assertEquals(MISMATCH, result["Issue_organisation_code"])
    }

    @Test
    fun internalNamesThroughTheOrder279Table() {
        // Ч -> 3, Ш -> 4 follow the order's table; one changed letter must break it
        for ((name, wrong) in listOf("Last_name_ru" to "ЛЕЩКИН", "First_name_ru" to "ПАВЕК", "Middle_name_ru" to "ПЕТРОВИ")) {
            assertEquals(MISMATCH, statuses(intOcr + (name to wrong), "INTPASSPORT_2011")[name], name)
        }
    }

    @Test
    fun yoAndYeAreTheSameLetterForTheZone() {
        val ocr = intOcr + ("MRZ" to intMrz.replace("LE4KIN", "L24KIN"))
        assertEquals(VERIFIED, statuses(ocr, "INTPASSPORT_2011")["Last_name_ru"])
        assertEquals(VERIFIED, statuses(ocr + ("Last_name_ru" to "ЛЁШКИН"), "INTPASSPORT_2011")["Last_name_ru"])
    }

    @Test
    fun sexNegativeControl() {
        assertEquals(MISMATCH, statuses(intOcr + ("Sex_ru" to "ЖЕН."), "INTPASSPORT_2011")["Sex_ru"])
    }

    @Test
    fun failedCheckDigitTakesTheZonePartOutOfPlay() {
        val zone = intMrz.replace("RUS8704234", "RUS8704244") // birth date, digit kept
        val result = checks(intOcr + ("MRZ" to zone), "INTPASSPORT_2011")
        assertEquals(MISMATCH, result.getValue("MRZ").status)
        assertTrue("MRZ-3" !in result.getValue("Birth_date").checks.map { it.id })
        assertEquals(UNCHECKED, result.getValue("Birth_date").status)
        assertEquals(VERIFIED, result.getValue("Licence_number").status) // other parts still hold
    }

    @Test
    fun malformedZoneIsAMismatchAndNoReference() {
        val result = checks(intOcr + ("MRZ" to intMrz.dropLast(1)), "INTPASSPORT_2011")
        assertEquals(MISMATCH, result.getValue("MRZ").status)
        assertEquals(UNCHECKED, result.getValue("Licence_number").status)
    }

    @Test
    fun line1WithAnImpossibleCharacterIsNoReference() {
        val zone = extMrz.replace("ROGOVA", "R0GOVA")
        val result = checks(extOcrConsistent + ("MRZ" to zone), "EXTPASSPORTBIO_2007")
        assertEquals(MISMATCH, result.getValue("MRZ").status)
        assertTrue(("MRZ-9" to "mismatch") in idsOf(result.getValue("MRZ")))
        assertEquals(UNCHECKED, result.getValue("Last_name_en").status)
        assertEquals(VERIFIED, result.getValue("Licence_number").status) // line 2 still holds
    }

    @Test
    fun internalZoneDigitsOutsideTheTableAreImpossible() {
        for (wrong in listOf("LE0KIN", "LE1KIN", "LE5KIN")) {
            val result = checks(intOcr + ("MRZ" to intMrz.replace("LE4KIN", wrong)), "INTPASSPORT_2011")
            assertEquals(MISMATCH, result.getValue("MRZ").status, wrong)
            assertEquals(UNCHECKED, result.getValue("Last_name_ru").status, wrong)
        }
    }

    @Test
    fun zoneIsVerifiedOnlyThroughItsNames() {
        val names = setOf("Last_name_ru", "First_name_ru", "Middle_name_ru")
        val noNames = intOcr.filterKeys { it !in names }
        assertEquals(UNCHECKED, statuses(noNames, "INTPASSPORT_2011")["MRZ"])
        assertEquals(VERIFIED, statuses(intOcr, "INTPASSPORT_2011")["MRZ"])
        // a page name that disagrees withholds the confirmation, it does not blame the zone
        val result = statuses(intOcr + ("First_name_ru" to "ПАВЕК"), "INTPASSPORT_2011")
        assertEquals(UNCHECKED, result["MRZ"])
        assertEquals(MISMATCH, result["First_name_ru"])
    }

    @Test
    fun zoneLayoutFollowsTheClassifierNotTheZoneTypeCharacters() {
        // 'PH' also fails the line-1 form (MRZ-9), so names are not compared; the optional data of line 2 is
        // still read with the internal layout
        val zone = "PH" + intMrz.substring(2)
        assertEquals(VERIFIED, statuses(intOcr + ("MRZ" to zone), "INTPASSPORT_2011")["Issue_date"])
    }

    @Test
    fun digitsAgreeingDoNotHideStrayCharacters() {
        val result = statuses(
            intOcr + ("Licence_number" to "88 12 106\"436\"\"") + ("Issue_organisation_code" to "770--021"),
            "INTPASSPORT_2011",
        )
        assertEquals(MISMATCH, result["Licence_number"])
        assertEquals(MISMATCH, result["Issue_organisation_code"])
        // whitespace is not part of the form
        assertEquals(
            VERIFIED,
            statuses(intOcr + ("Licence_number" to "45  07 318294"), "INTPASSPORT_2011")["Licence_number"],
        )
    }

    @Test
    fun externalNumberForm() {
        assertEquals(
            MISMATCH,
            statuses(extOcrConsistent + ("Licence_number" to "764820157"), "EXTPASSPORTBIO_2007")["Licence_number"],
        )
    }

    // --- external passport zone ----------------------------------------------

    @Test
    fun consistentExternalPassport() {
        val result = statuses(extOcrConsistent + ("Issue_date" to "20.05.2021"), "EXTPASSPORTBIO_2007")
        assertEquals(setOf(VERIFIED), result.values.toSet())
    }

    @Test
    fun externalNegativeControls() {
        val ocr = extOcrConsistent + ("Expiration_date" to "20.05.2032") + ("Last_name_en" to "ROGOVA") +
            ("First_name_en" to "IRENA")
        val result = statuses(ocr, "EXTPASSPORTBIO_2007")
        assertEquals(MISMATCH, result["Expiration_date"])
        assertEquals(MISMATCH, result["First_name_en"])
    }

    @Test
    fun theSpecimensAlteredNumberIsAMismatch() {
        assertEquals(MISMATCH, statuses(extOcr, "EXTPASSPORTBIO_2007")["Licence_number"])
    }

    /** A TD3 second line with every check digit computed, for building cases. */
    private fun td3Line2(number: String, birth: String, sex: String, expiry: String, personal: String = ""): String {
        fun digit(s: String) = MrzChecks.checkDigit(s).toString()
        val num = number.padEnd(9, '<')
        val per = personal.padEnd(14, '<')
        val line = num + digit(num) + "RUS" + birth + digit(birth) + sex + expiry + digit(expiry) + per + digit(per)
        val composite = line.substring(0, 10) + line.substring(13, 20) + line.substring(21, 43)
        return line + digit(composite)
    }

    @Test
    fun td3BuilderReproducesARealZone() {
        assertEquals(extMrz.lines()[1], td3Line2("764820157", "880314", "F", "310520"))
    }

    @Test
    fun sevenDigitZoneNumberOfThe2003Form() {
        val zone = extMrz.lines()[0] + "\n" + td3Line2("4820157", "880314", "F", "310520")
        val ocr = extOcrConsistent + ("MRZ" to zone)
        val result = checks(ocr, "EXTPASSPORT_2003")
        assertEquals(VERIFIED, result.getValue("MRZ").status)
        assertTrue(("MRZ-2" to "ok") in idsOf(result.getValue("Licence_number")))
        val bad = checks(ocr + ("Licence_number" to "76 4820158"), "EXTPASSPORT_2003").getValue("Licence_number")
        assertEquals(MISMATCH, bad.status)
    }

    @Test
    fun truncatedNameComparesByPrefix() {
        val longName = "KONSTANTINOPOLSKAYA".repeat(3)
        val zone = ("P<RUS" + "ROGOVA<<" + longName).take(44) + "\n" + extMrz.lines()[1]
        val ocr = extOcrConsistent + ("MRZ" to zone) + ("First_name_en" to longName)
        assertEquals(VERIFIED, statuses(ocr, "EXTPASSPORTBIO_2007")["First_name_en"])
    }

    // --- dates ---------------------------------------------------------------

    private val dlOcr = mapOf(
        "Birth_date" to "01.02.1990", "Issue_date" to "15.03.2015",
        "Expiration_date" to "15.03.2025", "Licence_number" to "99 12 345678",
    )

    @Test
    fun licenceValidityConfirmsBothDates() {
        val result = statuses(dlOcr, "DL_2011")
        assertEquals(VERIFIED, result["Issue_date"])
        assertEquals(VERIFIED, result["Expiration_date"])
        assertEquals(UNCHECKED, result["Birth_date"]) // only refuting checks reach it
        assertEquals(UNCHECKED, result["Licence_number"])
    }

    @Test
    fun licenceValidityNegativeControlMarksBothWithReferences() {
        val result = checks(dlOcr + ("Expiration_date" to "16.03.2025"), "DL_2020")
        assertEquals(MISMATCH, result.getValue("Issue_date").status)
        assertEquals(MISMATCH, result.getValue("Expiration_date").status)
        assertEquals("15.03.2025", result.getValue("Expiration_date").checks.first { it.id == "DL-2" }.reference)
        assertEquals("16.03.2015", result.getValue("Issue_date").checks.first { it.id == "DL-2" }.reference)
    }

    @Test
    fun externalValidityAcceptsFiveOrTenYears() {
        for (years in listOf(5, 10)) {
            val ocr = mapOf("Issue_date" to "01.06.2010", "Expiration_date" to "01.06.${2010 + years}")
            assertEquals(VERIFIED, statuses(ocr, "EXTPASSPORT_2003")["Issue_date"])
        }
        val ocr = mapOf("Issue_date" to "01.06.2010", "Expiration_date" to "01.06.2017")
        assertEquals(MISMATCH, statuses(ocr, "EXTPASSPORTBIO_2007")["Issue_date"])
        // the external passport has no single expected date, so no reference
        val dates = checks(ocr, "EXTPASSPORTBIO_2007").getValue("Issue_date").checks
        assertNull(dates.first { it.id == "EXT-3" }.reference)
    }

    @Test
    fun leapDayValidity() {
        val ocr = mapOf("Issue_date" to "29.02.2016", "Expiration_date" to "28.02.2026")
        assertEquals(VERIFIED, statuses(ocr, "DL_2011")["Expiration_date"])
        // both candidates exist for 29.02, so no single reference to show
        val check = checks(ocr, "DL_2011").getValue("Expiration_date").checks.first { it.id == "DL-2" }
        assertNull(check.reference)
    }

    @Test
    fun notACalendarDate() {
        assertEquals(MISMATCH, statuses(mapOf("Birth_date" to "31.02.1990"), "SNILS_1996")["Birth_date"])
        assertEquals(UNCHECKED, statuses(mapOf("Birth_date" to "10 ДЕКАБРЯ 1999 ГОДА"), "SNILS_1996")["Birth_date"])
    }

    @Test
    fun issueBeforeBirthMarksBoth() {
        val result = statuses(mapOf("Birth_date" to "01.01.2000", "Issue_date" to "01.01.1999"), "SNILS_1996")
        assertEquals(mapOf("Birth_date" to MISMATCH, "Issue_date" to MISMATCH), result)
    }

    @Test
    fun futureDate() {
        assertEquals(MISMATCH, statuses(mapOf("Issue_date" to "01.01.2030"), "SNILS_1996")["Issue_date"])
    }

    @Test
    fun minimumAgeAtIssue() {
        val young = mapOf("Birth_date" to "01.01.2000", "Issue_date" to "31.12.2013")
        assertEquals(MISMATCH, statuses(young, "INTPASSPORT_1997")["Issue_date"])
        assertEquals(UNCHECKED, statuses(young + ("Issue_date" to "01.01.2014"), "INTPASSPORT_1997")["Issue_date"])
        assertEquals(
            MISMATCH,
            statuses(mapOf("Birth_date" to "01.01.2000", "Issue_date" to "01.06.2015"), "DL_2020")["Birth_date"],
        )
        // SNILS is issued to children: no age rule
        assertEquals(UNCHECKED, statuses(young, "SNILS_1996")["Issue_date"])
    }

    // --- SNILS ---------------------------------------------------------------

    @Test
    fun snilsCheckNumber() {
        // the worked example of docs/validation-checks.md
        assertEquals(VERIFIED, statuses(mapOf("Licence_number" to "112-233-445 95"), "SNILS_1996")["Licence_number"])
        val verdict = checks(mapOf("Licence_number" to "112-233-445 96"), "SNILS_1996").getValue("Licence_number")
        assertEquals(MISMATCH, verdict.status)
        assertEquals("95", verdict.checks.last().reference)
        // a changed digit in the body
        assertEquals(MISMATCH, statuses(mapOf("Licence_number" to "112-233-446 95"), "SNILS_1996")["Licence_number"])
    }

    @Test
    fun snilsCheckNumberWrapsMod101() {
        // all nines: 9 * (9+8+...+1) = 405 -> 405 mod 101 = 1
        assertEquals(VERIFIED, statuses(mapOf("Licence_number" to "999-999-999 01"), "SNILS_1996")["Licence_number"])
        // 9*9 + 2*8 + 3*1 = 100 -> 00, not 100
        assertEquals(VERIFIED, statuses(mapOf("Licence_number" to "920-000-003 00"), "SNILS_1996")["Licence_number"])
        assertEquals(MISMATCH, statuses(mapOf("Licence_number" to "920-000-003 01"), "SNILS_1996")["Licence_number"])
    }

    @Test
    fun snilsLengthAndSmallNumbers() {
        assertEquals(MISMATCH, statuses(mapOf("Licence_number" to "112-233-445 9"), "SNILS_1996")["Licence_number"])
        assertEquals(MISMATCH, statuses(mapOf("Licence_number" to "112-233-445-95"), "SNILS_1996")["Licence_number"])
        // up to 001-001-998 there is no valid check number to verify
        assertEquals(UNCHECKED, statuses(mapOf("Licence_number" to "001-001-998 00"), "SNILS_1996")["Licence_number"])
    }

    // --- soft ----------------------------------------------------------------

    @Test
    fun seriesYearIsANoteNotAVerdict() {
        val ocr = intOcr + ("Licence_number" to "45 02 318294")
        val notes = checks(ocr, "INTPASSPORT_1997").getValue("Licence_number").notes
        assertEquals("PASS-3", notes.first().id)
        assertEquals("series printed in 2002, issued in 2007", notes.first().text)
        // the note does not change any verdict by itself
        val plain = checks(ocr + ("MRZ" to ""), "INTPASSPORT_1997").getValue("Licence_number")
        assertEquals(UNCHECKED, plain.status)
        assertEquals("PASS-3", plain.notes.first().id)
    }

    @Test
    fun seriesYearInsideTheWindowIsNoNote() {
        val verdict = checks(intOcr, "INTPASSPORT_2011").getValue("Licence_number")
        assertTrue(verdict.notes.isEmpty())
        assertNotNull(verdict.checks.firstOrNull())
    }

    @Test
    fun resultsWithoutReadingHaveNoVerdicts() {
        Results().use { assertEquals(emptyMap(), it.fieldChecks) }
    }

    // --- the stage's JSON ----------------------------------------------------

    @Test
    fun jsonOmitsReferenceWhenThereIsNone() {
        val json = FieldChecks.toJson(checks(dlOcr, "DL_2011")).toString()
        assertTrue("\"reference\":\"15.03.2025\"" in json)
        val birth = FieldChecks.toJson(checks(dlOcr, "DL_2011")).getValue("Birth_date").toString()
        assertTrue("reference" !in birth)
        assertEquals(
            "{\"status\":\"unchecked\",\"checks\":[],\"notes\":[]}",
            FieldChecks.toJson(checks(mapOf("Sex_ru" to ""), "DL_2011")).getValue("Sex_ru").toString(),
        )
    }
}
