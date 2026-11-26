program test_forest_fractional_samples
 use sklearn_common,only:dp,require
 use sklearn_ensemble,only:random_forest_regressor,extra_trees_regressor,random_forest_classifier, &
  extra_trees_classifier,reset_forest_fit
 implicit none
 type(random_forest_regressor)::reg,integer_reg
 type(extra_trees_regressor)::extra
 type(random_forest_classifier)::cls
 type(extra_trees_classifier)::ec
 real(dp)::x(9,2),y(9)
 integer::labels(9),i,n,k,expected
 integer,allocatable::bags(:,:)
 do i=1,9
  x(i,:)=[real(i,dp),real(mod(i,3),dp)];y(i)=real(i*i,dp);labels(i)=mod(i,2)
 end do
 reg%n_estimators=3;integer_reg%n_estimators=3;cls%n_estimators=3;extra%n_estimators=3;ec%n_estimators=3
 call extra%set_bootstrap(.true.);call ec%set_bootstrap(.true.)
 do n=5,9,2
  expected=4;if(n==5)expected=2
  call reg%set_max_samples(.5_dp);call cls%set_max_samples(.5_dp)
  call extra%set_max_samples(.5_dp);call ec%set_max_samples(.5_dp)
  call integer_reg%set_max_samples(expected);call integer_reg%fit(x(:n,:),y(:n))
  call reg%fit(x(:n,:),y(:n));bags=reg%estimators_samples()
  call require(size(bags,1)==expected,'fraction uses ties-to-even rounding')
  call require(all(bags==integer_reg%estimators_samples()),'fraction preserves seeded bootstrap draws')
  call cls%fit(x(:n,:),labels(:n));bags=cls%estimators_samples()
  call require(size(bags,1)==expected,'classification fraction')
  call extra%fit(x(:n,:),y(:n));bags=extra%estimators_samples()
  call require(size(bags,1)==expected,'ExtraTrees regression fraction')
  call ec%fit(x(:n,:),labels(:n));bags=ec%estimators_samples()
  call require(size(bags,1)==expected,'ExtraTrees classification fraction')
 end do
 do k=1,4
  select case(k)
  case(1);call reg%set_max_samples(.001_dp);expected=1
  case(2);call reg%set_max_samples(1._dp);expected=9
  case(3);call reg%set_max_samples(1);expected=1
  case(4);call reg%set_max_samples(0);expected=9
  end select
  call reset_forest_fit(reg);call reg%fit(x,y);bags=reg%estimators_samples()
  call require(size(bags,1)==expected,'typed setters and clone reset retain requested sample mode')
 end do
 print *,'Fractional forest draw counts, ties, seeded samples, integer reset and clone parameters: PASS'
end program
