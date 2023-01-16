import myokit.formats

i = myokit.formats.importer('cellml')
mod = i.model('./BPS2020_2.cellml')

myokit.save('BPS_2.mmt', mod)
