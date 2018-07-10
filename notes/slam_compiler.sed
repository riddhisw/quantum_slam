s/\\crefrangeconjunction/ to\\nobreakspace /g 
s/\\crefrangepreconjunction//g 
s/\\crefrangepostconjunction//g 
s/\\crefpairconjunction/ and\\nobreakspace /g 
s/\\crefmiddleconjunction/, /g 
s/\\creflastconjunction/ and\\nobreakspace /g 
s/\\crefpairgroupconjunction/ and\\nobreakspace /g 
s/\\crefmiddlegroupconjunction/, /g 
s/\\creflastgroupconjunction/, and\\nobreakspace /g 
s/\\cref@line@name /line/g 
s/\\cref@line@name@plural /lines/g 
s/\\Cref@line@name /Line/g 
s/\\Cref@line@name@plural /Lines/g 
s/\\cref@listing@name /listing/g 
s/\\cref@listing@name@plural /listings/g 
s/\\Cref@listing@name /Listing/g 
s/\\Cref@listing@name@plural /Listings/g 
s/\\cref@algorithm@name /algorithm/g 
s/\\cref@algorithm@name@plural /algorithms/g 
s/\\Cref@algorithm@name /Algorithm/g 
s/\\Cref@algorithm@name@plural /Algorithms/g 
s/\\cref@note@name /note/g 
s/\\cref@note@name@plural /notes/g 
s/\\Cref@note@name /Note/g 
s/\\Cref@note@name@plural /Notes/g 
s/\\cref@remark@name /remark/g 
s/\\cref@remark@name@plural /remarks/g 
s/\\Cref@remark@name /Remark/g 
s/\\Cref@remark@name@plural /Remarks/g 
s/\\cref@example@name /example/g 
s/\\cref@example@name@plural /examples/g 
s/\\Cref@example@name /Example/g 
s/\\Cref@example@name@plural /Examples/g 
s/\\cref@result@name /result/g 
s/\\cref@result@name@plural /results/g 
s/\\Cref@result@name /Result/g 
s/\\Cref@result@name@plural /Results/g 
s/\\cref@definition@name /definition/g 
s/\\cref@definition@name@plural /definitions/g 
s/\\Cref@definition@name /Definition/g 
s/\\Cref@definition@name@plural /Definitions/g 
s/\\cref@proposition@name /proposition/g 
s/\\cref@proposition@name@plural /propositions/g 
s/\\Cref@proposition@name /Proposition/g 
s/\\Cref@proposition@name@plural /Propositions/g 
s/\\cref@corollary@name /corollary/g 
s/\\cref@corollary@name@plural /corollaries/g 
s/\\Cref@corollary@name /Corollary/g 
s/\\Cref@corollary@name@plural /Corollaries/g 
s/\\cref@footnote@name /footnote/g 
s/\\cref@footnote@name@plural /footnotes/g 
s/\\Cref@footnote@name /Footnote/g 
s/\\Cref@footnote@name@plural /Footnotes/g 
s/\\cref@enumi@name /item/g 
s/\\cref@enumi@name@plural /items/g 
s/\\Cref@enumi@name /Item/g 
s/\\Cref@enumi@name@plural /Items/g 
s/\\cref@part@name /part/g 
s/\\cref@part@name@plural /parts/g 
s/\\Cref@part@name /Part/g 
s/\\Cref@part@name@plural /Parts/g 
s/\\cref@page@name /page/g 
s/\\cref@page@name@plural /pages/g 
s/\\Cref@page@name /Page/g 
s/\\Cref@page@name@plural /Pages/g 
s/\\cref@lemma@name /Lemma/g 
s/\\cref@lemma@name@plural /Lemmas/g 
s/\\Cref@lemma@name /\\MakeUppercase Lemma/g 
s/\\Cref@lemma@name@plural /\\MakeUppercase Lemmas/g 
s/\\cref@theorem@name /Theorem/g 
s/\\cref@theorem@name@plural /Theorems/g 
s/\\Cref@theorem@name /\\MakeUppercase Theorem/g 
s/\\Cref@theorem@name@plural /\\MakeUppercase Theorems/g 
s/\\cref@appendix@name /Appendix/g 
s/\\cref@appendix@name@plural /Appendices/g 
s/\\Cref@appendix@name /\\MakeUppercase Appendix/g 
s/\\Cref@appendix@name@plural /\\MakeUppercase Appendices/g 
s/\\cref@chapter@name /Chapter/g 
s/\\cref@chapter@name@plural /Chapters/g 
s/\\Cref@chapter@name /\\MakeUppercase Chapter/g 
s/\\Cref@chapter@name@plural /\\MakeUppercase Chapters/g 
s/\\cref@section@name /Section/g 
s/\\cref@section@name@plural /Sections/g 
s/\\Cref@section@name /\\MakeUppercase Section/g 
s/\\Cref@section@name@plural /\\MakeUppercase Sections/g 
s/\\cref@table@name /Table/g 
s/\\cref@table@name@plural /Tables/g 
s/\\Cref@table@name /\\MakeUppercase Table/g 
s/\\Cref@table@name@plural /\\MakeUppercase Tables/g 
s/\\cref@figure@name /Fig\./g 
s/\\cref@figure@name@plural /Figs\./g 
s/\\Cref@figure@name /\\MakeUppercase Fig\./g 
s/\\Cref@figure@name@plural /\\MakeUppercase Figs\./g 
s/\\cref@equation@name /Eq\./g 
s/\\cref@equation@name@plural /Eqs\./g 
s/\\Cref@equation@name /\\MakeUppercase Eq\./g 
s/\\Cref@equation@name@plural /\\MakeUppercase Eqs\./g 
s/\\label\[[^]]*\]/\\label/g
s/\\usepackage\(\[.*\]\)\{0,1\}{cleveref}//g
s/\\[cC]refformat{.*}{.*}//g
s/\\[cC]refrangeformat{.*}{.*}//g
s/\\[cC]refmultiformat{.*}{.*}{.*}{.*}//g
s/\\[cC]refrangemultiformat{.*}{.*}{.*}{.*}//g
s/\\[cC]refname{.*}{.*}//g
s/\\[cC]reflabelformat{.*}{.*}//g
s/\\[cC]refrangelabelformat{.*}{.*}//g
s/\\[cC]refdefaultlabelformat{.*}//g
s/\\renewcommand{\\crefpairconjunction}{.*}//g
s/\\renewcommand{\\crefpairgroupconjunction}{.*}//g
s/\\renewcommand{\\crefmiddleconjunction}{.*}//g
s/\\renewcommand{\\crefmiddlegroupconjunction}{.*}//g
s/\\renewcommand{\\creflastconjunction}{.*}//g
s/\\renewcommand{\\creflastgroupconjunction}{.*}//g
s/\\renewcommand{\\[cC]ref}{.*}//g
s/\\renewcommand{\\[cC]refrange}{.*}//g

