program anova
  use sklearn,only:dp,f_oneway,anova_sample,anova_result
  implicit none
  type(anova_sample)::groups(3)
  type(anova_result)::test
  groups(1)=anova_sample(reshape([1._dp,2._dp,3._dp],[3,1]))
  groups(2)=anova_sample(reshape([2._dp,3._dp,4._dp,5._dp],[4,1]))
  groups(3)=anova_sample(reshape([4._dp,6._dp],[2,1]))
  test=f_oneway(groups)
  print *,'F statistic:',test%statistic
  print *,'p value:',test%pvalue
end program
