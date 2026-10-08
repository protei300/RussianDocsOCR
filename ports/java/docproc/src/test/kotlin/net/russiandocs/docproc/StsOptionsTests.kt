package net.russiandocs.docproc

import net.russiandocs.docproc.modules.OcrCorrections
import net.russiandocs.docproc.pipeline.OcrOptions
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * OCR options for the vehicle registration certificate (issue #17) — the cases of `tests/test_sts_options.py`.
 *
 * One options class serves both sides of the document (STS_<year> is the vehicle side, STSBACK_<year> the owner
 * side): the dispatcher is given the type and matches the substring 'sts', which both share. Every field the
 * detector can produce for either side has to be routed to an engine, or the pipeline detects it and silently
 * drops it.
 */
class StsOptionsTests {

    private val front = listOf("Reg_number", "VIN", "Vehicle_make_ru", "Vehicle_make_en", "Vehicle_type",
        "Vehicle_category", "Vehicle_year", "Chassis_number", "Body_number", "Vehicle_color", "Engine_power",
        "Eco_class", "Max_mass", "Curb_mass", "Expiration_date", "PTS_number", "Licence_number")
    private val back = listOf("Licence_number", "Last_name_ru", "Last_name_en", "First_name_ru", "First_name_en",
        "Middle_name_ru", "Living_region_ru", "House_number", "Apartment_number", "Special_marks",
        "Issue_organisation_code", "Issue_date")
    private val newForm = listOf("Type_approval", "Building_number")
    private val old2010 = listOf("Engine_model", "Engine_number", "Engine_volume", "Issue_organization_ru",
        "Issue_date", "Building_number")

    private val sts = OcrOptions.forDocType("STS_1996")

    @Test
    fun bothSidesReachTheSameOptions() {
        for (t in listOf("STS", "STS_1996", "STSBACK", "STSBACK_1996", "STS_2019", "STSBACK_2019")) {
            assertEquals(sts, OcrOptions.forDocType(t), t)
        }
    }

    @Test
    fun noOtherTypeIsCaughtByTheStsBranch() {
        for (t in listOf("INTPASSPORT_2011", "INTPASSPORTADDR_ALL", "EXTPASSPORT_2003", "DL_2011", "SNILS_1996",
            "BIRTHCERT_2018")) {
            assertTrue(OcrOptions.forDocType(t) != sts, t)
        }
    }

    @Test
    fun theLicenceBackSideGetsEmptyOptionsNotTheFrontSideOnes() {
        // 'dlback' contains 'dl': without its own branch the back side would look for name fields on a table.
        for (t in listOf("DLBACK", "DLBACK_ALL")) {
            val o = OcrOptions.forDocType(t)
            assertTrue(o.ruFields.isEmpty() && o.enFields.isEmpty(), t)
        }
        assertTrue(OcrOptions.forDocType("DL_2011").enFields.isNotEmpty())
    }

    @Test
    fun theYearSuffixSplitsOffCleanly() {
        assertEquals("STSBACK" to "1996", OcrOptions.splitDocType("STSBACK_1996"))
    }

    @Test
    fun everyFieldOfEachSideIsRoutedToAnEngine() {
        for ((name, fields) in mapOf("front" to front, "back" to back, "new" to newForm, "old2010" to old2010)) {
            val missing = fields.filter { !sts.isOcrField(it) }
            assertTrue(missing.isEmpty(), "$name: detected but never read: $missing")
        }
    }

    @Test
    fun noFieldIsRoutedToBothEngines() {
        assertEquals(emptySet(), sts.ruFields.toSet() intersect sts.enFields.toSet())
    }

    @Test
    fun regNumberAndVinAreLatin() {
        // Decision of 2026-09-05: plate letters and VIN are Latin in the ground truth and on the output.
        assertTrue("Reg_number" in sts.enFields && "VIN" in sts.enFields)
    }

    @Test
    fun theSeriesKeepsThePassportPrecedent() {
        assertTrue("Licence_number" in sts.ruFields)
    }

    @Test
    fun theVehicleModelLineIsGone() {
        // a7b12e81: the field detector v10 has no such class
        assertFalse("Vehicle_model_en" in sts.enFields || "Vehicle_model_en" in sts.neededSplit)
    }

    @Test
    fun aVinNeverHoldsTheLetterO() {
        assertEquals("WF0DXXGAJD1A00001", OcrCorrections.checkVin("WFODXXGAJD1A00001"))
        assertEquals("XTA00000000123456", OcrCorrections.checkVin("XTAOOOOOOOO123456"))
        assertEquals("WF0DXXGAJD1A00001", OcrCorrections.checkVin("WF0DXXGAJD1A00001"))
    }

    @Test
    fun theMakeEngineFollowsTheStsForm() {
        assertTrue("Vehicle_make_ru" in sts.ruFields)           // the old form's route
        assertEquals(setOf("Special_marks"), sts.readMargin.keys)
        assertEquals("lat", sts.engineByYear["Vehicle_make_ru"]?.get("2019"))
        assertEquals(null, sts.engineByYear["Vehicle_make_ru"]?.get("1996"))
        assertEquals(null, sts.engineByYear["Vehicle_color"])
        assertEquals(listOf("Special_marks"), sts.glueTorn)
        // and no other type has such rules
        assertTrue(OcrOptions.forDocType("INTPASSPORT_2011").engineByYear.isEmpty())
        assertTrue(OcrOptions.forDocType("INTPASSPORT_2011").readMargin.isEmpty())
        assertTrue(OcrOptions.forDocType("INTPASSPORT_2011").glueTorn.isEmpty())
    }
}
